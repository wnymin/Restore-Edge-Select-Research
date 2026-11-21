"""Separate local forecast loss from certified full-trace offline headroom."""
import json
from math import prod

from scripts.validation.study_r03_estimate import Trigger, ROOT, read, F, jsonable


def evaluate():
    model = json.loads((ROOT/'configs/qwen3.5-4b.json').read_text(encoding='utf-8-sig'))['models'][0]
    rows = []
    for case in read('r02-cpu-inputs.json')['cases']:
        if case['id'].split('-')[2] != '+0':
            continue
        exp = Trigger(case['trace'], model, F(case['capacity_bytes'], 1 << 20))
        exact = exp.offline()
        optimum = exact['response_mean_ms']
        settings = {'nominal': {}, 'ssd_half': {'ssd_to_device': F(1,2)}, 'ssd_double': {'ssd_to_device': F(2)},
                    'compute_half': {'compute': F(1,2)}, 'compute_double': {'compute': F(2)}}
        scores = {}
        for name, estimate in settings.items():
            run, _ = exp.run_trigger(estimate, 'full')
            scores[name] = dict(mean=run['response_mean_ms'], actions=run['boundaries_tokens'],
                                gap_to_optimum=run['response_mean_ms']-optimum)
            assert scores[name]['gap_to_optimum'] >= 0
        assert exp.run('fixed', boundaries=exact['boundaries_tokens'])['response_mean_ms'] == optimum
        rows.append(dict(id=case['id'], split=case['split'], search_space=prod(map(len,exp.eligible)),
                         optimum=optimum, optimal_plan=exact['boundaries_tokens'], scores=scores))
    return jsonable(rows)


if __name__ == '__main__':
    print(json.dumps(evaluate(), indent=2))
