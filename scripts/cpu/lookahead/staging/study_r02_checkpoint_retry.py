"""Shared checkpoint admission at an exact byte threshold after release."""
import argparse
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path

from scripts.cpu.lookahead.staging.study_lookahead_dynamic_release import Capture, ROOT, make_trace, read
from scripts.cpu.lookahead.staging.study_lookahead_two_request import jsonable


def study():
    trace = make_trace()
    trace['requests'][3] = dict(deepcopy(trace['requests'][2]), id='D')
    model = read(ROOT / 'configs/qwen3.5-4b.json')['models'][0]
    plan = [0, 576, 1152, 1152]
    exp = Capture(trace, model, 0)
    threshold = exp.size(set(exp.candidates[2][1152]) - exp.initial)
    rows = []
    for delta in (-1, 0, 1):
        exp = Capture(trace, model, Fraction(threshold + delta, 1 << 20))
        result = exp.run('fixed', boundaries=plan)
        reservations = result['prefetch_reserved_at_ms']
        checkpoint_loads = [row for row in result['transfers'] if row['object'] == 'C.state2']
        expected = 30 if delta >= 0 else None
        assert reservations == [None, 0, expected, expected]
        assert len(checkpoint_loads) == (1 if delta >= 0 else 2)
        assert all(row['mode'] == ('prefetch' if delta >= 0 else 'demand') for row in checkpoint_loads)
        rows.append(dict(case=f'r02-checkpoint-retry-{delta:+d}', delta_bytes=delta, capacity_bytes=exp.budget,
                         reservations=reservations, completions=result['completion_ms'],
                         transfers=result['transfers']))
    return jsonable(dict(trace=trace, threshold_bytes=threshold, plan=plan, cases=rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args()
    result = study()
    if args.check:
        assert read(args.check) == result
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('Verified three shared checkpoint release cases.')
