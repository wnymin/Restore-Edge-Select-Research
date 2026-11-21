"""Compare existing coarse screening and target-only forecasts to full windows."""
from fractions import Fraction as F
import json

from scripts.validation.study_r03_estimate import Trigger, ROOT, read, jsonable


def evaluate():
    model = json.loads((ROOT/'configs/qwen3.5-4b.json').read_text(encoding='utf-8-sig'))['models'][0]
    rows = []
    for case in read('r02-cpu-inputs.json')['cases']:
        if case['id'].split('-')[2] != '+0':
            continue
        exp = Trigger(case['trace'], model, F(case['capacity_bytes'], 1 << 20))
        reference, _ = exp.run_trigger({}, 'full')
        for mode in ('legacy', 'target'):
            run, _ = exp.run_trigger({}, mode)
            fixed = exp.run('fixed', boundaries=run['boundaries_tokens'])
            assert fixed['completion_ms'] == run['completion_ms']
            a, b = [[t-arrival for t, arrival in zip(r['completion_ms'], exp.arrivals, strict=True)]
                    for r in (reference, run)]
            rows.append(dict(id=case['id'], split=case['split'], mode=mode,
                reference_actions=reference['boundaries_tokens'], actions=run['boundaries_tokens'],
                reference_responses=a, responses=b, changed=reference['boundaries_tokens']!=run['boundaries_tokens'],
                mean_loss_ms=(sum(b)-sum(a))/len(a), mean_loss_fraction=(sum(b)-sum(a))/sum(a),
                p95_loss_ms=max(b)-max(a), max_request_loss_ms=max(F(0),*(y-x for x,y in zip(a,b,strict=True))),
                reference_replays=reference['decision_replays'], replays=run['decision_replays'],
                reference_events=reference['decision_events'], events=run['decision_events']))
    return jsonable(rows)


if __name__ == '__main__':
    print(json.dumps(evaluate(), indent=2))
