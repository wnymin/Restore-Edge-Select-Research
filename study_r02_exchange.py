"""Export a versioned CPU exchange table with observations and outcomes separated."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path

from scripts.cpu.lookahead.staging.study_r02_checkpoint_retry import study as checkpoint
from scripts.cpu.lookahead.staging.study_r02_reference import Capture, ROOT, read, jsonable


def export():
    witness = checkpoint()
    trace = witness['trace']
    model = read(ROOT / 'configs/qwen3.5-4b.json')['models'][0]
    observations, outcomes = [], []
    for case in witness['cases']:
        exp = Capture(trace, model, F(case['capacity_bytes']) / (1 << 20))
        plans = ([0, 576, 1152, 1152], [0, 1152, 1152, 1152])
        runs = [exp.run('fixed', boundaries=plan) for plan in plans]
        observations.append(dict(
            id=case['case'], source='docs/evidence/r02-cpu-checkpoint-retry.json',
            candidates=[576, 1152], fixed_successors=[1152, 1152],
            capacity_bytes=int(exp.budget), head='A', target='B',
            visible=['A', 'B', 'C', 'D'], decision_ms='0',
            clock='Exact simulator milliseconds since common arrival',
            initial_ready=sorted(exp.initial), reservations={}, inflight=None,
            active_head_remaining_ms=['30', '30'], observation_status='known synthetic state',
            alias_objects=sorted(set(exp.candidates[2][1152]) & set(exp.candidates[3][1152])),
        ))
        shallow, deep = runs
        assert jsonable(shallow['completion_ms']) == case['completions']
        delta = deep['response_sum_ms'] - shallow['response_sum_ms']
        outcomes.append(dict(
            id=case['case'], completion_ms=[r['completion_ms'] for r in runs],
            reservation_ms=[r['prefetch_reserved_at_ms'] for r in runs],
            transfers=[r['transfers'] for r in runs], deep_minus_shallow_sum_ms=delta,
            best=1152 if delta <= 0 else 576, tied=delta == 0,
        ))
    return jsonable(dict(schema=1, base='3458577b8f43a9286b1763eac5c56025ea866197',
                         scope='Synthetic CPU fixed plans from identical first-entry state',
                         observations=observations, outcomes=outcomes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args()
    data = export()
    if args.check:
        assert read(args.check) == data
    if args.out:
        args.out.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8', newline='\n')
    print([(r['id'], r['deep_minus_shallow_sum_ms'], r['best']) for r in data['outcomes']])

