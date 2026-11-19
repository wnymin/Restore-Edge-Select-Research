"""Compare suffix coalescence with complete online window evaluation."""
import argparse
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
from random import Random
from statistics import median

from scripts.cpu.lookahead.screening.study_lookahead_coalescence import Coalescence, Trigger
from scripts.cpu.lookahead.staging.study_r02_reference import ROOT, read, jsonable


from scripts.cpu.lookahead.staging.study_r02_guarded import Guarded


def study(repeats, original_arrivals=False):
    model = read(ROOT / 'configs/qwen3.5-4b.json')['models'][0]
    rng = Random(72844)
    rows = []
    for case in read(ROOT / 'docs/evidence/r02-cpu-inputs.json')['cases']:
        trace = deepcopy(case['trace'])
        if not original_arrivals:
            for request in trace['requests']:
                request['arrival_ms'] = 0
        capacity = F(case['capacity_bytes'], 1 << 20)
        base = Trigger(trace, model, capacity)
        adapter = Guarded if original_arrivals else Coalescence
        candidate = adapter(trace, model, capacity, audit=True)
        candidate.project = True
        expected, _ = base.run_trigger({}, 'full')
        audited, _ = candidate.run_trigger({}, 'full')
        for prop in ('completion_ms', 'boundaries_tokens'):
            assert audited[prop] == expected[prop], (case['id'], prop)
        candidate.audit = False
        reverse = rng.choice((False, True))
        for inject in (False, True):
            for rep in range(repeats):
                order = [base, candidate]
                if (rep + inject) % 2:
                    order.reverse()
                # Pair direction is balanced across repetitions.
                if reverse:
                    order.reverse()
                values = {}
                for exp in order:
                    result, wall = exp.run_trigger({}, 'full', measure=inject, profile=not inject)
                    if not inject:
                        for prop in ('completion_ms', 'boundaries_tokens'):
                            assert result[prop] == expected[prop], (case['id'], prop)
                    response = [t - a for t, a in zip(result['completion_ms'], exp.arrivals, strict=True)]
                    values[exp is candidate] = (result, wall, response)
                old, new = values[False], values[True]
                rows.append(dict(id=case['id'], split=case['split'], inject=inject, rep=rep,
                                 reference_first=order[0] is base, reference_wall_ms=old[1], projected_wall_ms=new[1],
                                 reference_decision_ns=sum(ns for _, ns in old[0]['decision_profile_ns']),
                                 projected_decision_ns=sum(ns for _, ns in new[0]['decision_profile_ns']),
                                 reference_completions=old[0]['completion_ms'], projected_completions=new[0]['completion_ms'],
                                 reference_boundaries=old[0]['boundaries_tokens'], projected_boundaries=new[0]['boundaries_tokens'],
                                 net_mean_ms=(sum(old[2]) - sum(new[2])) / len(new[2]),
                                 net_p95_ms=max(old[2]) - max(new[2]),
                                 fallback=candidate.stats['guard_fallback'],
                                 max_harm_ms=max(0, *(b-a for a,b in zip(old[2],new[2],strict=True))),
                                 hits=candidate.stats['merged_forecasts'],
                                 forecasts=new[0]['decision_replays']))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--original-arrivals', action='store_true')
    args = parser.parse_args()
    if args.repeats < 2 or args.repeats % 2:
        parser.error('Use a positive even repeat count')
    rows = study(args.repeats, args.original_arrivals)
    args.out.write_text(json.dumps(jsonable(rows), separators=(',', ':')) + '\n', encoding='utf-8')
    for split in ('design', 'holdout'):
        timed = [r for r in rows if r['split'] == split and not r['inject']]
        injected = [r for r in rows if r['split'] == split and r['inject']]
        print(split, 'median paired decision saving ns',
              median(r['reference_decision_ns']-r['projected_decision_ns'] for r in timed),
              'median net response ms', float(median(r['net_mean_ms'] for r in injected)),
              'max individual harm ms', float(max(r['max_harm_ms'] for r in injected)),
              'hits', sum(r['hits'] for r in timed if r['rep'] == 0))


