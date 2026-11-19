"""Freeze R02 inputs and evaluate fixed plans with the original executor."""
import argparse
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
from random import Random

from scripts.cpu.lookahead.staging.study_lookahead_dynamic_release import make_trace
from scripts.cpu.lookahead.staging.study_lookahead_three_request import Capture, ROOT, read
from scripts.cpu.lookahead.staging.study_lookahead_two_request import jsonable


def freeze():
    model = read(ROOT / 'configs/qwen3.5-4b.json')['models'][0]
    rows = []
    for split, seed, speed in [('design', 72841, (1, 2, 3)), ('holdout', 72842, (4, 5, 7))]:
        rng = Random(seed)
        for sharing in ('none', 'kv', 'checkpoint'):
            for delta in (-1, 0, 1):
                for timing in (-1, 0, 1):
                    trace = make_trace()
                    b, c = trace['requests'][1:3]
                    if sharing != 'none':
                        c['input_token_runs'] = deepcopy(b['input_token_runs'])
                        deep = deepcopy(b['candidates'][-1])
                        if sharing == 'kv':
                            deep['load_order'][-1] = 'C.state2'
                            obj = next(o for o in trace['objects'] if o['id'] == 'C.state2')
                            obj['prefix_token_runs'] = deepcopy(b['input_token_runs'])
                        c['candidates'] = [c['candidates'][0], deep]
                        trace['cost']['device_copy'] = dict(startup_ms=0, bytes_per_ms=100000000)
                    c['arrival_ms'] = str(30 + F(timing, 8))
                    for path in ('ram_to_device', 'ssd_to_device'):
                        cfg = trace['cost'][path]
                        cfg['bytes_per_ms'] = str(F(cfg['bytes_per_ms']) / rng.choice(speed))
                    exp = Capture(trace, model, 0)
                    # Capacity around simultaneous shallow B and deep C dependencies.
                    deps = set(exp.candidates[1][576]) | set(next(
                        q for q in trace['requests'] if q['id'] == 'C')['candidates'][-1]['load_order'])
                    capacity = exp.size(deps - exp.initial) + delta
                    rows.append(dict(id=f'{split}-{sharing}-{delta:+d}-{timing:+d}', split=split,
                                     seed=seed, capacity_bytes=capacity, trace=trace))
    return dict(schema=1, source_base='3458577b8f43a9286b1763eac5c56025ea866197', cases=rows)


def evaluate(frozen, split):
    model = read(ROOT / 'configs/qwen3.5-4b.json')['models'][0]
    rows = []
    for case in frozen['cases']:
        if case['split'] != split:
            continue
        exp = Capture(case['trace'], model, F(case['capacity_bytes'], 1 << 20))
        plans = []
        for boundary in (0, 576, 1152):
            plan = [dict(A=0, B=boundary, C=1152, D=576)[q['id']] for q in exp.requests]
            result = exp.run('fixed', boundaries=plan)
            plans.append(dict(boundary=boundary, plan=plan, completion_ms=result['completion_ms'],
                              response_sum_ms=result['response_sum_ms'],
                              reservation_ms=result['prefetch_reserved_at_ms']))
        winner = min(plans, key=lambda p: (p['response_sum_ms'], -p['boundary']))
        rows.append(dict(id=case['id'], order=[q['id'] for q in exp.requests],
                         winner=winner['boundary'], plans=plans))
    return jsonable(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--split', choices=('design', 'holdout'), default='design')
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.freeze:
        with args.inputs.open('x', encoding='utf-8') as stream:
            json.dump(freeze(), stream, indent=2)
        return
    rows = evaluate(read(args.inputs), args.split)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'Evaluated {len(rows)} {args.split} cases, {3 * len(rows)} complete plans')


if __name__ == '__main__':
    main()
