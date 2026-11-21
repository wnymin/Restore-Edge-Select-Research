"""Isolate cross-request object aliasing on fixed R02 anchor traces."""
from copy import deepcopy
from fractions import Fraction as F
import json

from scripts.validation.study_r03_capacity import read, replay
from scripts.validation.study_r03_visibility import ranking


def evaluate():
    output = {split: [] for split in ('design', 'holdout')}
    for source in read('r02-cpu-inputs.json')['cases']:
        _, layout, capacity, _ = source['id'].split('-', 3)
        if layout not in ('kv', 'checkpoint') or capacity != '+0':
            continue
        clone = deepcopy(source)
        trace = clone['trace']
        requests = {q['id']: q for q in trace['requests']}
        b_objects = {obj for plan in requests['B']['candidates'] for obj in plan['load_order']}
        c_plans = requests['C']['candidates']
        shared = {o for p in c_plans for o in p['load_order']} & b_objects
        objects = {o['id']: o for o in trace['objects']}
        assert shared and shared.isdisjoint(trace['scheduling']['initial_device_object_ids'])
        aliases = {obj: f'r03.independent.{obj}' for obj in sorted(shared)}
        assert set(aliases.values()).isdisjoint(objects)
        trace['objects'].extend(dict(deepcopy(objects[obj]), id=new) for obj, new in aliases.items())
        for plan in c_plans:
            plan['load_order'] = [aliases.get(o, o) for o in plan['load_order']]
        clone['id'] += '/independent'
        split = source['split']
        original, changed = replay({'cases': [source, clone]}, split)
        assert original['order'] == changed['order']
        arrival = sorted(F(q['arrival_ms']) for q in trace['requests'])
        plans = {p['boundary']: p for p in changed['plans']}
        best = plans[changed['winner']]
        stale = plans[original['winner']]
        v, w = [[F(t)-a for t,a in zip(p['completion_ms'],arrival,strict=True)] for p in (best,stale)]
        rank_a, ties_a = ranking(original['plans'])
        rank_b, ties_b = ranking(changed['plans'])
        output[split].append(dict(id=source['id'], layout=layout, aliases=aliases,
            before=original, after=changed, changed=original['winner']!=changed['winner'],
            rank_changed=rank_a!=rank_b, tie_set_changed=ties_a!=ties_b,
            mean_loss_ms=str((sum(w)-sum(v))/len(v)), mean_loss_fraction=str((sum(w)-sum(v))/sum(v)),
            p95_loss_ms=str(max(w)-max(v)), max_request_loss_ms=str(max(F(0),*(b-a for a,b in zip(v,w,strict=True))))))
    return output


if __name__ == '__main__':
    print(json.dumps(evaluate(), indent=2))

