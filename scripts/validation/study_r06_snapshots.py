"""Fixed-snapshot witnesses for R06, using every legal fixed continuation."""

from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import sys

from scripts.cpu.replay.study_r06_transfer import ANCHOR, MODELS, Experiment, ROOT, allow_serial, read, run, score
from scripts.validation.check_r04_snapshots import snapshot


def records():
    fixture = read(ROOT / 'fixtures/phase1-hand-trace.json')
    model = read(ROOT / MODELS[ANCHOR])['models'][0]

    def measure(label, trace, budget, previous):
        exp = Experiment(trace, model, budget)
        rows, serial, ends = snapshot(exp, previous)
        key = {'placement': 'ssd', 'ssd': '11'}
        chosen = {p: run(exp, key, p, {'margin': '1/100'}, overrides=dict(enumerate(previous)))
                  ['boundaries_tokens'][-1] for p in ('entry-replay', 'portfolio-v4', 'unified')}
        winner = min(ends, key=lambda c: (sum(ends[c]), -c))
        coverage = rows[serial]['head_slack_ms'] + rows[serial]['committed_ahead_compute_ms']
        no_coverage = min(rows, key=lambda c: (score(dict(rows[c], head_slack_ms=0,
                                                         committed_ahead_compute_ms=0)), -c))
        return dict(case=label, budget_mib=str(budget), previous=previous, serial=serial, exact=winner,
                    chosen=chosen, unified_fast=allow_serial(rows, serial, exp.budget, F('1/100')),
                    no_coverage_score_winner=no_coverage, coverage_ms=coverage,
                    outcomes={c: dict(ends=ends[c], loss_ms=sum(ends[c]) - sum(ends[winner]),
                                      projected_ms=score(r), snapshot=r) for c, r in rows.items()})

    # Same capacities, dependency graph and arrivals in each cost pair.
    for cost_scale in ('1', '2'):
        for arrival in ('26', '26749/1000', '107/4', '26751/1000', '30'):
            trace = deepcopy(fixture)
            trace['requests'] = trace['requests'][:2]
            trace['requests'][1]['arrival_ms'] = arrival
            path = trace['cost']['ram_to_device']
            path['bytes_per_ms'] = str(F(path['bytes_per_ms']) / F(cost_scale))
            for budget in (F('537/8') - F(1, 2**20), F('537/8'), F(128)):
                yield measure(f'coverage/ram-{cost_scale}/arrival-{arrival}', trace, budget, [0])

    for shared in (False, True):
        for arrival in ('1', '3/2', '25', '30'):
            trace = deepcopy(fixture)
            if shared:
                trace['cost']['device_copy'] = {'startup_ms': 0, 'bytes_per_ms': 51511296}
                trace['requests'][2] = dict(trace['requests'][1], id='C')
            trace['requests'][2]['arrival_ms'] = arrival
            yield measure(f'shared-{shared}/arrival-{arrival}', trace, 128, [0, 1152])


def payload():
    rows = list(records())
    at = next(r for r in rows if r['case'] == 'coverage/ram-1/arrival-107/4' and r['budget_mib'] == '537/8')
    left = next(r for r in rows if r['case'] == 'coverage/ram-1/arrival-26749/1000' and r['budget_mib'] == '537/8')
    right = next(r for r in rows if r['case'] == 'coverage/ram-1/arrival-26751/1000' and r['budget_mib'] == '537/8')
    below = next(r for r in rows if r['case'] == at['case'] and r['budget_mib'] not in {'128', '537/8'})
    assert at['outcomes'][576]['loss_ms'] == at['outcomes'][1152]['loss_ms'] == 0
    assert (left['exact'], at['exact'], right['exact']) == (576, 1152, 1152)
    assert not below['outcomes'][576]['snapshot']['prefetch_fits'] and at['outcomes'][576]['snapshot']['prefetch_fits']
    assert any(r['coverage_ms'] == 0 for r in rows)
    assert any(r['exact'] != r['no_coverage_score_winner'] for r in rows)
    for condition in ('ready_bytes', 'inflight_ms'):
        assert any(o['snapshot'][condition] and o['snapshot']['shared_reservation_bytes']
                   and not r['unified_fast'] for r in rows for c, o in r['outcomes'].items() if c)
    return json.loads(json.dumps(rows, default=str))


if __name__ == '__main__':
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'docs/evidence/r06-snapshots.json'
    path.write_text(json.dumps(payload(), indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {path}')
