"""Compare R02 suffix keys at the executor's post-completion checkpoints."""
import argparse
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path

from scripts.cpu.lookahead.screening.study_lookahead_workload_bounds import clone
from scripts.cpu.lookahead.staging.study_r02_reference import ROOT, read, jsonable
from scripts.cpu.lookahead.screening.study_lookahead_phase1 import Experiment


def evaluate(frozen, split):
    model = read(ROOT / 'configs/qwen3.5-4b.json')['models'][0]
    speeds = dict(ram_to_device=F(1), ssd_to_device=F(1))
    rows = []
    for case in frozen['cases']:
        if case['split'] != split:
            continue
        trace = deepcopy(case['trace'])
        # This variant starts from a common, entirely visible window.
        for request in trace['requests']:
            request['arrival_ms'] = 0
        exp = Experiment(trace, model, F(case['capacity_bytes'], 1 << 20))
        initial = (F(0), 0, 4, {}, set(exp.initial), None, None, (), set(), None, {})
        caches = {mode: {} for mode in ('full', 'projected', 'coarse')}
        for boundary in (0, 576, 1152):
            plan = (0, boundary, 1152, 576)

            def replay(state, stop=None):
                return exp._run('fixed', 0, plan, 100, 4, snapshot=clone(state),
                                forecast_speed=speeds, lean_rollout=True, pause_at_head=stop)

            full = replay(initial)['completion_ms']
            part = replay(initial, 2)
            state = part['snapshot']
            now, head, visible, admitted, ready, running, flight, _, held, release, host = state
            assert (head, visible, running, release) == (2, 4, None, None)
            assert not held and not host
            relevant = set().union(*(exp.candidates[j][plan[j]] for j in range(head, visible)))
            reservations = tuple((j, frozenset(deps)) for j, deps in sorted(admitted.items()))
            residual = None if flight is None else flight[0] - now
            obj = None if flight is None else flight[1]
            keys = {
                'full': (reservations, frozenset(ready), residual, obj),
                'projected': (reservations, frozenset(ready & relevant), residual,
                              obj if obj in relevant else None),
                'coarse': (tuple((j, exp.size(deps)) for j, deps in reservations),
                           exp.size(ready & relevant), residual),
            }
            true = tuple(t - now for t in full[head:])
            assert part['completion_ms'] + replay(state)['completion_ms'] == full
            for mode, key in keys.items():
                previous = caches[mode].get(key)
                if previous is not None and mode != 'coarse':
                    resumed = part['completion_ms'] + [now + elapsed for elapsed in previous]
                    assert resumed == full, (case['id'], boundary, mode)
                caches[mode].setdefault(key, true)
                rows.append(dict(id=case['id'], boundary=boundary, representation=mode,
                                 hit=previous is not None, wrong=previous is not None and previous != true,
                                 ready=sorted(ready), flight=flight, prefix=part['completion_ms'],
                                 suffix_ms=true, predicted_suffix_ms=previous, complete_ms=full,
                                 reserved={j: sorted(deps) for j, deps in reservations}))
    return jsonable(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('design', 'holdout'), default='design')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = evaluate(read(ROOT / 'docs/evidence/r02-cpu-inputs.json'), args.split)
    args.out.write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8', newline='\n')
    for mode in ('full', 'projected', 'coarse'):
        group = [row for row in rows if row['representation'] == mode]
        print(mode, len(group), sum(row['hit'] for row in group), sum(row['wrong'] for row in group))


