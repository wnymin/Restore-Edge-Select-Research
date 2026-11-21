"""Replay R03 capacity, SSD cost and release interventions on frozen R02 anchors."""
import argparse
from copy import deepcopy
import json
from fractions import Fraction as F
from pathlib import Path

from scripts.cpu.lookahead.staging.study_r02_reference import evaluate as replay

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'docs/evidence'


def read(name):
    return json.loads((EVIDENCE / name).read_text(encoding='utf-8-sig'))


def replay_release(case, order, model):
    from restore_edge_select.simulator import Experiment

    exp = Experiment(case['trace'], model, F(case['capacity_bytes'], 1 << 20))
    plans = []
    for boundary in (0, 576, 1152):
        plan = [dict(A=0, B=boundary, C=1152, D=576)[request] for request in order]
        run = exp.run('fixed', boundaries=plan)
        plans.append(dict(boundary=boundary, response_sum_ms=run['response_sum_ms'],
                          completion_ms=run['completion_ms']))
    return dict(order=[q['id'] for q in exp.requests], plans=plans,
                winner=min(plans, key=lambda p: (p['response_sum_ms'], -p['boundary']))['boundary'])


def evaluate(factor="capacity"):
    output = {}
    for split in ('design', 'holdout'):
        source = read('r02-cpu-inputs.json')['cases']
        contrasts = []
        for a in source:
            if a['split'] != split or a['id'].split('-')[2] != '+0':
                continue
            levels = {'capacity': (-1, 1), 'ssd': (F(1, 2), F(2)), 'release': ('finish',)}[factor]
            for level in levels:
                if factor == 'capacity':
                    b = dict(a, id=f"{a['id']}/capacity{level:+d}", capacity_bytes=a['capacity_bytes'] + level)
                    perturbation = dict(delta_bytes=level)
                elif factor == 'ssd':
                    b = deepcopy(a)
                    b['id'] = f"{a['id']}/ssd{level}"
                    cost = b['trace']['cost']['ssd_to_device']
                    cost['bytes_per_ms'] = str(F(cost['bytes_per_ms']) / level)
                    perturbation = dict(ssd_payload_time_multiplier=str(level))
                else:
                    b = deepcopy(a)
                    b['id'] = f"{a['id']}/release-{level}"
                    schedule = b['trace']['scheduling']
                    assert schedule.get('prefetch_release', 'start') == 'start'
                    schedule['prefetch_release'] = level
                    perturbation = dict(release=level)
                if factor == 'release':
                    reference, = replay({'cases': [a]}, split)
                    model = json.loads((ROOT / 'configs/qwen3.5-4b.json').read_text(encoding='utf-8-sig'))['models'][0]
                    altered = replay_release(b, reference['order'], model)
                else:
                    reference, altered = replay({'cases': [a, b]}, split)
                assert reference['order'] == altered['order']
                chosen = {p['boundary']: p for p in altered['plans']}
                best, stale = chosen[altered['winner']], chosen[reference['winner']]
                arrival = sorted(F(q['arrival_ms']) for q in b['trace']['requests'])
                vr, vs = ([F(t)-a for t, a in zip(p['completion_ms'], arrival, strict=True)]
                          for p in (best, stale))
                mean = sum(vr) / len(vr)
                loss = (sum(vs)-sum(vr))/len(vr)
                ties = lambda row: sorted(p['boundary'] for p in row['plans']
                                          if F(p['response_sum_ms']) == min(F(x['response_sum_ms']) for x in row['plans']))
                rank = lambda row: sorted((p['boundary'] for p in row['plans']),
                    key=lambda c: (next(F(p['response_sum_ms']) for p in row['plans'] if p['boundary']==c), -c))
                contrasts.append(dict(reference=a['id'], altered=b['id'], **perturbation,
                    changed=reference['winner'] != altered['winner'], rank_changed=rank(reference)!=rank(altered),
                    tie_set_changed=ties(reference)!=ties(altered), legal_set_changed=False,
                    mean_loss_ms=str(loss), mean_loss_fraction=str(loss/mean),
                    p95_loss_ms=str(max(vs)-max(vr)), max_request_loss_ms=str(max(F(0),*(s-r for r,s in zip(vr,vs,strict=True))))))
        output[split] = dict(eligible=len(contrasts), pairs=contrasts)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", choices=("capacity", "ssd", "release"), default="capacity")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.factor), indent=2))




