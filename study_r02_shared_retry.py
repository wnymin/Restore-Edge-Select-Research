"""Check shared object identity during admission and release retries."""
import json
from copy import deepcopy
from scripts.cpu.lookahead.staging.study_r02_reference import Capture, ROOT, read, jsonable


def study():
    model = read(ROOT / 'configs/qwen3.5-4b.json')['models'][0]
    inputs = read(ROOT / 'docs/evidence/r02-cpu-inputs.json')['cases']
    rows = []
    for split in ('design', 'holdout'):
        for sharing in ('kv', 'checkpoint'):
            source = f'{split}-{sharing}-+0-+0'
            trace = deepcopy(next(case['trace'] for case in inputs if case['id'] == source))
            for request in trace['requests']:
                request['arrival_ms'] = 0
            for boundary in (576, 1152):
                for capacity in (100, 130, 145):
                    exp = Capture(trace, model, capacity)
                    plan = (0, boundary, 1152, 576)
                    result = exp.run('fixed', boundaries=plan)
                    reserved = result['prefetch_reserved_at_ms']
                    transfers = result['transfers']
                    starts = [a + b + c for a, b, c in zip(exp.arrivals, result['queue_wait_ms'],
                                                           result['head_to_compute_ms'], strict=True)]
                    shared = sorted(set(exp.candidates[1][boundary]) & set(exp.candidates[2][1152]))
                    kv_loads = sum(row['object'] == 'B.kv1' for row in transfers)
                    if split == 'design' and boundary == 576 and capacity == 100:
                        assert reserved[2] == starts[1] and reserved[2] > 0
                        assert kv_loads == 1
                    if split == 'design' and boundary == 576 and capacity == 130:
                        assert reserved[2] is None and kv_loads == 2
                    if sharing == 'checkpoint' and boundary == 1152:
                        assert 'B.state2' in shared
                        assert reserved[1] == reserved[2] == 0
                        assert sum(row['object'] == 'B.state2' for row in transfers) == 1
                    rows.append(dict(source=source, plan=plan, capacity_bytes=exp.budget,
                                     shared_objects=shared, reservation_ms=reserved, start_ms=starts,
                                     completion_ms=result['completion_ms'], transfers=transfers))
    return jsonable(rows)


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args()
    rows = study()
    if args.check:
        assert read(args.check) == rows
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'Checked {len(rows)} shared fixed-plan runs')

