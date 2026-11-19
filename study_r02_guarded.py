"""Verify guarded coalescence with original, staggered arrivals."""
import argparse
import json
from fractions import Fraction as F
from pathlib import Path

from scripts.cpu.lookahead.screening.study_lookahead_coalescence import Coalescence, Experiment, Trigger
from scripts.cpu.lookahead.staging.study_r02_reference import ROOT, jsonable, read


class Guarded(Coalescence):
    def _run(self, policy, threshold_tokens, boundaries, max_combinations, n, **kwargs):
        state = kwargs.get('snapshot')
        if state is not None:
            # A later request outside the forecast is excluded by the existing
            # visible-window contract, just as in the complete reference.
            if state[2] != n or self.arrivals[n-1] > state[0] or self.i + 1 >= n:
                self.stats['guard_fallback'] += 1
                return Experiment._run(self, policy, threshold_tokens, boundaries,
                                       max_combinations, n, **kwargs)
        return super()._run(policy, threshold_tokens, boundaries, max_combinations, n, **kwargs)


def study():
    model = read(ROOT / 'configs/qwen3.5-4b.json')['models'][0]
    rows = []
    for case in read(ROOT / 'docs/evidence/r02-cpu-inputs.json')['cases']:
        capacity = F(case['capacity_bytes'], 1 << 20)
        reference = Trigger(case['trace'], model, capacity)
        projected = Guarded(case['trace'], model, capacity, audit=True)
        projected.project = True
        baseline, _ = reference.run_trigger({}, 'full')
        audited, _ = projected.run_trigger({}, 'full')
        assert projected.stats['guard_fallback'] + projected.stats['audit_forecasts'] == audited['decision_replays']
        projected.audit = False
        result, _ = projected.run_trigger({}, 'full')
        for field in ('boundaries_tokens', 'completion_ms'):
            assert result[field] == audited[field] == baseline[field], (case['id'], field)
        rows.append(dict(id=case['id'], split=case['split'],
                         boundaries=result['boundaries_tokens'], completions=result['completion_ms'],
                         forecasts=result['decision_replays'], fallback=projected.stats['guard_fallback'],
                         hits=projected.stats['merged_forecasts'],
                         complete_events=baseline['decision_events'], guarded_events=result['decision_events']))
    return jsonable(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args()
    rows = study()
    if args.check:
        assert rows == read(args.check)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8', newline='\n')
    for split in ('design', 'holdout'):
        group = [r for r in rows if r['split'] == split]
        print(split, {k: sum(r[k] for r in group) for k in
                     ('forecasts', 'fallback', 'hits', 'complete_events', 'guarded_events')})

