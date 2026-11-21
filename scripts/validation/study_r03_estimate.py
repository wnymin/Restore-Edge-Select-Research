"""Closed-loop estimate error under unchanged actual execution costs."""
import argparse
from fractions import Fraction as F
import json

from scripts.cpu.lookahead.screening.study_lookahead_trigger import Trigger
from scripts.cpu.lookahead.staging.study_r02_reference import jsonable
from scripts.validation.study_r03_capacity import ROOT, read


def evaluate(path="ssd_to_device", companion=None):
    model = json.loads((ROOT/'configs/qwen3.5-4b.json').read_text(encoding='utf-8-sig'))['models'][0]
    rows = []
    for case in read('r02-cpu-inputs.json')['cases']:
        if case['id'].split('-')[2] != '+0':
            continue
        exp = Trigger(case['trace'], model, F(case['capacity_bytes'], 1 << 20))
        base, _ = exp.run_trigger({}, 'full')
        for multiplier in (F(1, 2), F(2)):
            estimates = {path: multiplier}
            if companion is not None:
                estimates['compute'] = companion
            run, _ = exp.run_trigger(estimates, 'full')
            assert exp.run('fixed', boundaries=run['boundaries_tokens'])['completion_ms'] == run['completion_ms']
            before, after = [[t-a for t,a in zip(r['completion_ms'],exp.arrivals,strict=True)] for r in (base,run)]
            delta = sum(after)-sum(before)
            rows.append(dict(id=case['id'], split=case['split'], multiplier=str(multiplier),
                             baseline_actions=base['boundaries_tokens'], estimate_actions=run['boundaries_tokens'],
                             baseline_responses=before, estimate_responses=after,
                             changed=base['boundaries_tokens']!=run['boundaries_tokens'],
                             mean_loss_ms=delta/len(before), mean_loss_fraction=delta/sum(before),
                             p95_loss_ms=max(after)-max(before),
                             max_request_loss_ms=max(F(0),*(b-a for a,b in zip(before,after,strict=True)))))
    return jsonable(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', choices=('ssd_to_device', 'compute'), default='ssd_to_device')
    args = parser.parse_args()
    print(json.dumps(evaluate(args.path), indent=2))



