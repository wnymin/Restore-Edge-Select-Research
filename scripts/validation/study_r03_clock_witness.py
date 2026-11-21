"""Replay the estimate timing witness with exact per-decision clock increments."""
from itertools import cycle
import json
from unittest.mock import patch

from scripts.validation.study_r03_estimate import Trigger, ROOT, read, F, jsonable


def evaluate():
    case = next(c for c in read('r02-cpu-inputs.json')['cases'] if c['id']=='holdout-checkpoint-+0-+0')
    model = json.loads((ROOT/'configs/qwen3.5-4b.json').read_text(encoding='utf-8-sig'))['models'][0]
    exp = Trigger(case['trace'], model, F(case['capacity_bytes'], 1 << 20))
    rows = []
    for delay in (0, 1, 1000, 10000, 100000, 1000000):
        for scale in (F(1), F(1,2)):
            ticks = cycle((0, delay))
            with patch('restore_edge_select.simulator.perf_counter_ns', side_effect=lambda: next(ticks)):
                r, _ = exp.run_trigger({'ssd_to_device': scale} if scale != 1 else {}, 'full', measure=True)
            assert all(ns==delay for _,ns in r['decision_profile_ns'])
            rows.append(dict(delay_ns=delay, estimate=str(scale), actions=r['boundaries_tokens'],
                             completions_ms=r['completion_ms'], reservation_ms=r['prefetch_reserved_at_ms'],
                             transfers=r['transfers']))
    return jsonable(dict(case=case['id'], order=[q['id'] for q in exp.requests], rows=rows))


if __name__ == '__main__':
    print(json.dumps(evaluate(), indent=2))
