"""Exact end-of-transfer ties at the release-and-retry event in witness 7884."""
import argparse
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path

from scripts.cpu.lookahead.staging.study_lookahead_dynamic_release import make_trace, Capture, ROOT, read
from scripts.cpu.lookahead.staging.study_lookahead_two_request import jsonable


def study():
    model = read(ROOT / 'configs/qwen3.5-4b.json')['models'][0]
    plan = [0, 576, 1152, 576]
    baseline = Capture(make_trace(), model, 145)
    needed = (set(baseline.candidates[2][1152]) | set(baseline.candidates[3][576])) - baseline.initial
    threshold = baseline.size(needed)
    rows = []
    for byte_delta, time_delta in product((-1, 0, 1), (F(-1, 8), F(0), F(1, 8))):
        trace = make_trace()
        checkpoint_bytes = baseline.objects['D.state1']['bytes']
        trace['cost']['ssd_to_device']['bytes_per_ms'] = str(F(2 * checkpoint_bytes) / (29 + time_delta))
        exp = Capture(trace, model, F(threshold + byte_delta, 1 << 20))
        result = exp.run('fixed', boundaries=plan)
        a_end = result['completion_ms'][0]
        b_start = exp.arrivals[1] + result['queue_wait_ms'][1] + result['head_to_compute_ms'][1]
        d_io = next(row for row in result['transfers'] if row['object'] == 'D.state1')
        c_io = next(row for row in result['transfers'] if row['object'] == 'C.kv1')
        reserved = result['prefetch_reserved_at_ms']
        assert reserved[1] == reserved[3] == 0
        assert a_end == b_start == 30
        assert d_io['end_ms'] - b_start == time_delta
        if byte_delta >= 0:
            assert reserved[2] == b_start
            assert c_io['mode'] == 'prefetch'
            assert c_io['start_ms'] == max(b_start, d_io['end_ms'])
        else:
            assert reserved[2] is None and c_io['mode'] == 'demand'
            assert c_io['start_ms'] == result['completion_ms'][1]
        rows.append(dict(id=f'r02-release-{byte_delta:+d}-{time_delta}', plan=plan,
                         bytes=threshold + byte_delta, threshold_bytes=threshold,
                         byte_delta=byte_delta, ready_minus_release_ms=time_delta,
                         release_ms=b_start, d_ready_ms=d_io['end_ms'], c_first_transfer=c_io,
                         reservation_ms=reserved, completion_ms=result['completion_ms']))
    return jsonable(dict(source='dynamic-lookahead-dynamic-release-7884', cases=rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args()
    result = study()
    if args.check:
        assert result == read(args.check)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f"Verified {len(result['cases'])} release and readiness cases")

