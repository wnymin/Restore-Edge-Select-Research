"""Three-request closed forms with a fixed, disjoint-object successor.

The active head computes; target and fixed successor may compete for prefetch
capacity. No active transfer, extra reservations, or sharing of missing objects.
Exact critical scenarios replace a full symbolic event tree in this submodel.
"""

import argparse
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import combinations
import json
from pathlib import Path
from random import Random

from scripts.cpu.lookahead.robustness.study_lookahead_joint_domain import Capture, Experiment, Linear, ROOT, Region, analyze, oracle, read
from scripts.cpu.lookahead.staging.study_lookahead_two_request import jsonable


@dataclass(frozen=True)
class Profile:
    load: Linear
    compute: F
    missing: frozenset


def profile(exp, i, c, estimate, speeds):
    deps = exp.candidates[i][c]
    assert len(deps) == len(set(deps))
    alpha = beta = F(0)
    missing = frozenset(deps)-exp.initial
    for obj in deps:
        if obj in exp.initial:
            continue
        path = {'RAM': 'ram_to_device', 'SSD': 'ssd_to_device'}[exp.objects[obj]['tier']]
        duration = exp._transfer_times[obj]*estimate.get(path,1)/speeds[path]
        assert duration >= 0
        if path == 'ssd_to_device':
            beta += duration
        else:
            alpha += duration
    suffix = exp._suffix_times[i][c]*estimate.get('compute',1)
    if deps and deps[-1] in exp.initial and c < exp.lengths[i]:
        suffix += exp.path_time('device_copy',exp.objects[deps[-1]]['bytes'])*estimate.get('device_copy',1)
    assert alpha >= 0 and beta >= 0 and suffix >= 0
    return Profile(Linear(alpha,0,beta),suffix,missing)


class Projection:
    def __init__(self, exp, record):
        target, fallback, state, base, estimate, speeds = record
        now, head, n, admitted, ready, running, transfer = state[:7]
        assert (head,target,n) == (0,1,3) and len(base) == 3
        assert not exp.per_tier and not exp.dynamic and not exp.host_cache_bytes
        assert exp.release_at == 'start' and exp.window >= 2
        assert running is not None and running > now and transfer is None
        assert not admitted and ready == exp.initial
        assert not state[8] and state[9] is None and not state[10]
        assert max(exp.arrivals) <= now
        self.now, self.fallback = now, fallback
        self.follower = profile(exp,2,base[2],estimate,speeds)
        follower_bytes = exp.size(self.follower.missing)
        assert follower_bytes <= exp.budget
        assert not self.follower.missing.intersection(state[7][:-1])
        self.targets = {c: profile(exp,1,c,estimate,speeds) for c in exp.eligible[1]}
        self.coexist = {}
        for c, p in self.targets.items():
            own_bytes = exp.size(p.missing)
            assert own_bytes <= exp.budget
            assert not p.missing.intersection(state[7][:-1])
            assert not p.missing.intersection(self.follower.missing), 'missing-object sharing is outside this theorem'
            # Addition is exact only after the disjointness assertion above.
            self.coexist[c] = own_bytes+follower_bytes <= exp.budget

    def vector(self, c, point):
        r, _ = point
        p = self.targets[c]
        load, later = p.load.at(point), self.follower.load.at(point)
        start = max(r,load)
        target = start+p.compute
        follower_ready = (load if self.coexist[c] else start)+later
        finish = max(target,follower_ready)+self.follower.compute
        return [self.now+r,self.now+target,self.now+finish]

    def hinges(self, c):
        p = self.targets[c]
        r = Linear(0,1,0)
        lines = [r-p.load,self.follower.load-p.compute]
        if self.coexist[c]:
            lines.append(r-p.load-self.follower.load+p.compute)
        return lines


def normalized(line):
    lead = F(line.r or line.x)
    if not lead:
        return None
    return Linear(F(line.a)/lead,F(line.r)/lead,F(line.x)/lead)


def critical_points(lines, box):
    lines = tuple({d for d in map(normalized,lines) if d is not None})
    rlo,xlo = box.vertices[0]
    rhi,xhi = box.vertices[2]
    points = set(box.vertices)

    def add(r,x):
        if rlo <= r <= rhi and xlo <= x <= xhi:
            points.add((r,x))

    for d in lines:
        if d.x:
            for r in (rlo,rhi):
                add(r,-(d.a+d.r*r)/d.x)
        if d.r:
            for x in (xlo,xhi):
                add(-(d.a+d.x*x)/d.r,x)
    for a,b in combinations(lines,2):
        det = a.r*b.x-b.r*a.x
        if det:
            add((a.x*b.a-b.x*a.a)/det,(b.r*a.a-a.r*b.a)/det)
    return points


def envelopes(model, box):
    choices = tuple(model.targets)
    total = {(c,c):F(0) for c in choices}
    component = {(c,c):(F(0),)*3 for c in choices}
    max_points = scenarios = 0
    for a,b in combinations(choices,2):
        points = critical_points(model.hinges(a)+model.hinges(b),box)
        assert len(points) <= 31
        differences = [tuple(x-y for x,y in zip(model.vector(a,p),model.vector(b,p))) for p in points]
        total[a,b] = max(sum(v) for v in differences)
        total[b,a] = -min(sum(v) for v in differences)
        component[a,b] = tuple(max(v[i] for v in differences) for i in range(3))
        component[b,a] = tuple(-min(v[i] for v in differences) for i in range(3))
        max_points = max(max_points,len(points))
        scenarios += len(points)
    upper = {c:max(component[c,model.fallback]) for c in choices}
    regret = {c:max(total[c,b] for b in choices) for c in choices}
    safe = [c for c in choices if upper[c] <= 0]
    chosen = min(safe,key=lambda c:(regret[c],c!=model.fallback,-c))
    return dict(individual=upper,total={c:total[c,model.fallback] for c in choices},
                component={c:component[c,model.fallback] for c in choices},
                regret=regret,protected=chosen,max_pair_points=max_points,pair_scenarios=scenarios)


def at_points(model, points):
    choices = tuple(model.targets)
    components = {c:[] for c in choices}
    totals = {c:[] for c in choices}
    regret = {c:F(0) for c in choices}
    for point in points:
        vectors = {c:model.vector(c,point) for c in choices}
        scores = {c:sum(v) for c,v in vectors.items()}
        for c in choices:
            components[c].append(tuple(a-b for a,b in zip(vectors[c],vectors[model.fallback])))
            totals[c].append(scores[c]-scores[model.fallback])
            regret[c] = max(regret[c],scores[c]-min(scores.values()))
    component = {c:tuple(max(v[j] for v in values) for j in range(3)) for c,values in components.items()}
    upper = {c:max(v) for c,v in component.items()}
    safe = [c for c in choices if upper[c] <= 0]
    choice = min(safe,key=lambda c:(regret[c],c!=model.fallback,-c))
    return dict(component=component,individual=upper,total={c:max(v) for c,v in totals.items()},
                regret=regret,protected=choice)


def seed_trace():
    trace = read(ROOT/'fixtures/phase1-hand-trace.json')
    trace.pop('expected_by_prefetch_mib',None)
    trace['fixture_id'] = 'three-request-critical-scenarios'
    trace['purpose'] = 'Synthetic critical-scenario analysis, not measured hardware costs.'
    trace['scheduling']['future_window_requests'] = 2
    for request in trace['requests']:
        request['arrival_ms'] = 0
    return trace


def audit_trace(trace, capacity, fixed, *, keep_trace=False, baseline=None, box=None):
    model = read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    exp = Capture(trace,model,capacity)
    exp.run_trigger({},'full')
    first = exp.records[0]
    assert first[0] == 1
    base = (*first[3][:2],fixed)
    record = (*first[:3],base,*first[4:])
    if baseline is not None:
        assert baseline in exp.eligible[1]
        record = (record[0],baseline,*record[2:])
    projection = Projection(exp,record)
    residual = record[2][5]-record[2][0]
    box = box or Region.rectangle(residual/4,residual*F(7,4),F(1,4),F(7,4))
    exact = envelopes(projection,box)
    symbolic = analyze(exp,record,rectangle=box)
    for row in symbolic:
        c = row['candidate']
        assert F(row['individual_upper']) == exact['individual'][c],('individual',c,row,exact)
        assert F(row['total_upper']) == exact['total'][c],('total',c,row,exact)
        assert F(row['regret']) == exact['regret'][c],('regret',c,row,exact)
        assert row['protected'] == exact['protected']
    points = critical_points([d for c in projection.targets for d in projection.hinges(c)],box)
    global_values = at_points(projection,points)
    assert all(global_values[k] == exact[k] for k in global_values)
    reduced_points = critical_points([Linear(0,1,0)-p.load for p in projection.targets.values()],box)
    reduced = at_points(projection,reduced_points)
    corner = at_points(projection,box.vertices)
    diagonal = at_points(projection,(box.vertices[1],box.vertices[3]))
    reduced_gap = any(reduced[k] != exact[k] for k in ('individual','total','regret'))
    points |= {box.center(),(residual,F(1))}
    vectors = []
    for point in sorted(points):
        current = {}
        for c in projection.targets:
            value = oracle(exp,record[2],(0,c,fixed),{},record[5],point)
            assert value == projection.vector(c,point),('vector',c,point,value,projection.vector(c,point))
            current[c] = value
        vectors.append(dict(point=point,completion=current))
    result = dict(capacity_mib=capacity,fixed_successor=fixed,fallback=record[1],
                  profiles={c:dict(alpha=p.load.a,beta=p.load.x,compute=p.compute,
                                    missing=sorted(p.missing),coexist=projection.coexist[c])
                            for c,p in projection.targets.items()},
                  successor=dict(alpha=projection.follower.load.a,beta=projection.follower.load.x,
                                 compute=projection.follower.compute),
                  exact=exact,reduced=reduced,corner=corner,diagonal=diagonal,
                  target_only_points=len(reduced_points),reduced_gap=reduced_gap,
                  symbolic_rows=symbolic,critical_vectors=len(points)*len(projection.targets))
    if keep_trace:
        result.update(trace=trace,critical_checks=vectors)
    if baseline is not None:
        result['native_fallback'] = first[1]
    return jsonable(result)


def run_stress(seed=7790, rounds=64):
    rng = Random(seed)
    cases = []
    for index in range(rounds):
        trace = seed_trace()
        factor = rng.choice((F(1,4),F(1,2),F(1),F(2),F(4)))
        for key in ('a_ms_per_token_rational','b_ms_per_pair_rational'):
            a,b = trace['cost']['recompute'][key]
            value = F(a,b)*factor
            trace['cost']['recompute'][key] = [value.numerator,value.denominator]
        for obj in trace['objects']:
            obj['tier'] = rng.choice(('RAM','SSD'))
        trace['scheduling']['initial_device_object_ids'] = [o['id'] for o in trace['objects'] if rng.randrange(5)==0]
        trace['cost']['device_copy'] = dict(startup_ms='1/10',bytes_per_ms=str(1<<20))
        for path in ('ram_to_device','ssd_to_device'):
            trace['cost'][path] = dict(startup_ms=str(rng.randrange(4)),
                                     bytes_per_ms=str(F(16<<20,rng.randrange(1,17))))
        capacity = rng.choice((128,160,192,256))
        fixed = rng.choice((0,576,1152))
        result = audit_trace(trace,capacity,fixed)
        result['case'] = index
        cases.append(result)
    return dict(seed=seed,cases=cases,configurations=rounds,
                critical_vectors=sum(c['critical_vectors'] for c in cases),
                symbolic_vectors=sum(c['symbolic_rows'][0]['probes'] for c in cases),
                symbolic_cells=sum(c['symbolic_rows'][0]['cells'] for c in cases),
                mixed_capacity_cases=sum(len({p['coexist'] for p in c['profiles'].values()})>1 for c in cases),
                target_only_gap_cases=sum(c['reduced_gap'] for c in cases),
                max_pair_points=max(c['exact']['max_pair_points'] for c in cases))


def shared_counterexample():
    trace = seed_trace()
    for key in ('a_ms_per_token_rational','b_ms_per_pair_rational'):
        a,b = trace['cost']['recompute'][key]
        value = F(a,b)/5
        trace['cost']['recompute'][key] = [value.numerator,value.denominator]
    trace['requests'][2]['input_token_runs'][0]['token_id'] = 20
    trace['objects'] = [o for o in trace['objects'] if not o['id'].startswith('C.kv')]
    for obj in trace['objects']:
        if obj['id'].startswith('C.state'):
            obj['prefix_token_runs'][0]['token_id'] = 20
    for candidate in trace['requests'][2]['candidates']:
        candidate['load_order'] = [k.replace('C.kv','B.kv') for k in candidate['load_order']]
    model = read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    exp = Capture(trace,model,160)
    exp.run_trigger({},'full')
    first = exp.records[0]
    record = (*first[:3],(*first[3][:2],1152),*first[4:])
    try:
        Projection(exp,record)
    except AssertionError as error:
        assert 'missing-object sharing' in str(error)
    else:
        raise AssertionError('projection silently accepted shared objects')
    target = profile(exp,1,1152,{},record[5])
    follower = profile(exp,2,1152,{},record[5])
    assert target.load.at((F(6),F(1))) == follower.load.at((F(6),F(1))) == 5
    union = exp.size(target.missing|follower.missing)
    gross = exp.size(target.missing)+exp.size(follower.missing)
    assert union <= exp.budget < gross
    out = Experiment._run(exp,'fixed',0,(0,1152,1152),100,3)
    assert out['completion_ms'] == [F(6),F(6),F(8)]
    return jsonable(dict(trace=trace,capacity_mib=160,shared_bytes=gross-union,
                         union_bytes=union,gross_bytes=gross,actual_completion=out['completion_ms'],
                         naive_gross_completion=[6,6,11],union_only_completion=[6,6,10],
                         reason='Union accounting alone does not remove duplicated transfer work.'))


def threshold_counterexample():
    trace = seed_trace()
    trace['fixture_id'] = 'successor-consumption-threshold'
    for request in trace['requests'][1:]:
        request['arrival_ms'] = 29.25
    trace['scheduling']['initial_device_object_ids'] = ['B.kv1']
    for obj in trace['objects']:
        if obj['id'] == 'B.state2':
            obj['tier'] = 'RAM'
        elif obj['id'] != 'B.kv1':
            obj['tier'] = 'SSD'
    checkpoint = F(51511296)
    ssd_time = F(2227,908)
    trace['cost']['ram_to_device'] = dict(startup_ms=0,bytes_per_ms=str(checkpoint/F(19,5)))
    trace['cost']['ssd_to_device'] = dict(startup_ms=0,bytes_per_ms=str(checkpoint/ssd_time))
    box = Region.rectangle(F(1,2),F(1),F(1,2),F(3,2))
    result = audit_trace(trace,128,1152,keep_trace=True,baseline=576,box=box)
    full,reduced = result['exact'],result['reduced']
    assert full['total']['1152'] == full['regret']['1152'] == '1099/4540'
    assert reduced['total']['1152'] == '-2987/9080' and reduced['regret']['1152'] == '0'
    assert full['individual']['1152'] == '10197/4540'
    assert result['target_only_points'] == 4 and result['reduced_gap']
    model = read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    exp = Experiment(trace,model,128)
    actual = {c:Experiment._run(exp,'fixed',0,(0,c,1152),100,3)['completion_ms'] for c in exp.eligible[1]}
    delta = sum(actual[1152])-sum(actual[576])
    assert delta == F(1099,4540)
    def budget_choice(values,cap):
        safe = [int(c) for c,h in values['individual'].items() if F(h) <= cap]
        return min(safe,key=lambda c:(F(values['regret'][str(c)]),c!=576,-c))
    assert budget_choice(reduced,F(2)) == 1152 and budget_choice(full,F(2)) == 576
    changes = [a-b for a,b in zip(actual[1152],actual[576])]
    assert changes[2] == F(10197,4540) > 2
    result.update(native_fixed_plan_completion=jsonable(actual),native_deep_total_delta=str(delta),
                  r_domain=['1/2','1'],x_domain=['1/2','3/2'],
                  missing_surface='successor load equals shallow target compute: x=1',
                  budget_example=dict(budget_ms='2',reduced_choice=1152,exact_choice=576,
                                      actual_component_delta=jsonable(changes)))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds',type=int,default=64)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--check',type=Path)
    parser.add_argument('--threshold-out',type=Path)
    parser.add_argument('--threshold-check',type=Path)
    args = parser.parse_args()
    result = run_stress(rounds=args.rounds)
    result['shared_counterexample'] = shared_counterexample()
    if args.check:
        assert result == read(args.check)
        print('Closed forms, critical certificates, symbolic domains and native vectors reproduced.')
    if args.out:
        with args.out.open('x',encoding='utf-8') as f:
            json.dump(result,f,ensure_ascii=False,indent=2)
    if args.threshold_out or args.threshold_check:
        threshold = threshold_counterexample()
        if args.threshold_check:
            assert threshold == read(args.threshold_check)
        if args.threshold_out:
            with args.threshold_out.open('x',encoding='utf-8') as f:
                json.dump(threshold,f,ensure_ascii=False,indent=2)
        print('Successor threshold counterexample:',threshold['native_deep_total_delta'])
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','shared_counterexample')},sort_keys=True))
