"""Four-request capacity cascade with two fixed prefetch successors.

Research-only exact projection: B chooses a legal checkpoint; C and D are
fixed. B,C,D fit pairwise; their triple may fit or require D to wait for B
to release capacity. No claim for arbitrary queue or sharing structures.
"""

import argparse
from fractions import Fraction as F
import json
from pathlib import Path

from scripts.cpu.lookahead.robustness.study_lookahead_joint_domain import ROOT, read, Linear, Region, analyze, oracle
from scripts.cpu.lookahead.staging.study_lookahead_three_request import Capture, profile, seed_trace, critical_points
from scripts.cpu.lookahead.staging.study_lookahead_two_request import jsonable
from itertools import combinations
from random import Random


def make_trace():
    trace = seed_trace()
    trace['fixture_id'] = 'four-request-pairwise-fit-admission-cascade'
    trace['purpose'] = 'Synthetic exact event projection, not a hardware measurement.'
    trace['memory']['prefetch_mib_cases'] = [192]
    trace['scheduling']['future_window_requests'] = 3
    for request in trace['requests'][1:]:
        request['arrival_ms'] = 0
    trace['requests'][1]['candidates'] = [
        c for c in trace['requests'][1]['candidates']
        if c['boundary_tokens'] in (0, 576)]
    d = json.loads(json.dumps(trace['requests'][2]))
    d['id'] = 'D'
    d['input_token_runs'] = [{'token_id': 40, 'count': 1152}]
    for candidate in d['candidates']:
        candidate['load_order'] = [name.replace('C.', 'D.') for name in candidate['load_order']]
    trace['requests'].append(d)
    copies = []
    for obj in trace['objects']:
        if obj['id'].startswith('C.'):
            copy = json.loads(json.dumps(obj))
            copy['id'] = copy['id'].replace('C.', 'D.')
            for run in copy.get('prefix_token_runs', []):
                run['token_id'] = 40
            copies.append(copy)
    trace['objects'].extend(copies)
    return trace


def supports(b, c, d, coexist):
    """Max-affine supports for A,B,C,D, relative to snapshot time."""
    r = Linear(0, 1, 0)
    start = (r, b.load)
    target = tuple(v+b.compute for v in start)
    later = (*target, b.load+c.load)
    cfinish = tuple(v+c.compute for v in later)
    if coexist:
        dfinal = (*cfinish, b.load+c.load+d.load)
    else:
        dfinal = (*cfinish, r+d.load, b.load+c.load+d.load)
    return ((r,), target, cfinish, tuple(v+d.compute for v in dfinal))


def projected(table, point):
    return tuple(max(v.at(point) for v in forms) for forms in table)


def run(rounds=32):
    rng = Random(7880)
    cases = []
    for index in range(rounds):
        trace = make_trace()
        # Include all three target actions. Some capacities allow the shallow
        # triple but reject the deep triple, testing both admission branches.
        trace['requests'][1]['candidates'] = seed_trace()['requests'][1]['candidates']
        capacity = (180,192,240,256,280)[index % 5]
        trace['memory']['prefetch_mib_cases'] = [capacity]
        if index:
            trace['cost']['ram_to_device'] = dict(startup_ms=str(F(rng.randrange(0,4),4)),
                                                  bytes_per_ms=str(F(18<<20,rng.randrange(1,6))))
            trace['cost']['ssd_to_device'] = dict(startup_ms=str(F(rng.randrange(0,4),4)),
                                                  bytes_per_ms=str(F(51511296,rng.randrange(1,9))))
            for obj in trace['objects']:
                obj['tier'] = rng.choice(('RAM','SSD'))
            # Nonzero C,D suffixes exercise all max-affine supports; missing
            # sets retain the same deep checkpoints while prompts get longer.
            for request in trace['requests'][2:]:
                request['input_token_runs'][0]['count'] += rng.choice((0,128,576))
        model = read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
        exp = Capture(trace,model,capacity)
        exp.run_trigger({},'full')
        state = exp.records[0][2]
        assert state[0] == 0 and state[1] == 0 and state[2] == 4
        assert not state[3] and state[4] == exp.initial and state[6] is None
        speeds = {'ram_to_device':F(1),'ssd_to_device':F(1)}
        bp = {b:profile(exp,1,b,{},speeds) for b in exp.eligible[1]}
        cp,dp = profile(exp,2,1152,{},speeds),profile(exp,3,1152,{},speeds)
        missing = [p.missing for p in (*bp.values(),cp,dp)]
        for p in bp.values():
            assert not p.missing & (cp.missing | dp.missing)
            assert exp.size(p.missing | cp.missing) <= exp.budget
            assert exp.size(p.missing | dp.missing) <= exp.budget
        assert not cp.missing & dp.missing
        assert exp.size(cp.missing | dp.missing) <= exp.budget
        fits = {b:exp.size(p.missing | cp.missing | dp.missing) <= exp.budget
                for b,p in bp.items()}
        tables = {b:supports(p,cp,dp,fits[b]) for b,p in bp.items()}
        rnom = state[5]-state[0]
        box = Region.rectangle(rnom/8,rnom*F(7,4),F(1,4),F(7,4))
        lines = {b:[v-w for forms in tables[b] for v,w in combinations(forms,2)] for b in bp}
        pair_results = {}
        probes = set(box.vertices) | {box.center()}
        individual = {b:F(0) for b in bp}
        total = {}
        for b,a in combinations(bp,2):
            points = critical_points(lines[b]+lines[a],box)
            assert len(points) <= 438
            probes.update(points)
            deltas = [tuple(u-v for u,v in zip(projected(tables[b],p),projected(tables[a],p)))
                      for p in points]
            total[b,a] = max(map(sum,deltas))
            total[a,b] = -min(map(sum,deltas))
            if a == 0:
                individual[b] = max(v for delta in deltas for v in delta)
            if b == 0:
                individual[a] = max(-v for delta in deltas for v in delta)
            pair_results[f'{b}:{a}'] = dict(points=len(points),
                                          component_upper=[max(d[j] for d in deltas) for j in range(4)],
                                          total_upper=total[b,a],reverse_total_upper=total[a,b])
        for b in bp:
            total[b,b] = F(0)
        regret = {b:max(total[b,a] for a in bp) for b in bp}
        safe = [b for b in bp if individual[b] <= 0]
        protected = min(safe,key=lambda b:(regret[b],b!=0,-b))
        record = (1,0,state,(0,0,1152,1152),{},speeds)
        symbolic = analyze(exp,record,rectangle=box)
        for row in symbolic:
            b = row['candidate']
            assert F(row['individual_upper']) == individual[b]
            assert F(row['total_upper']) == total[b,0]
            assert F(row['regret']) == regret[b]
            assert row['protected'] == protected
        for point in probes:
            for b in bp:
                actual = oracle(exp,state,(0,b,1152,1152),{},speeds,point)
                assert tuple(actual) == projected(tables[b],point), (index,b,point,actual)
        if index == 0:
            point = (F(19,2),F(1))
            actual = oracle(exp,state,(0,576,1152,1152),{},speeds,point)
            assert actual == [F(19,2),F(55,4),F(55,4),F(29,2)]
            relaxed = Capture(trace,model,280)
            wide = oracle(relaxed,state,(0,576,1152,1152),{},speeds,point)
            assert wide == [F(19,2),F(55,4),F(55,4),F(14)]
        cases.append(jsonable(dict(index=index,capacity_mib=capacity,trace=trace,
                                   triple_fits=fits,profiles={b:dict(alpha=p.load.a,beta=p.load.x,compute=p.compute) for b,p in bp.items()},
                                   pairs=pair_results,regret=regret,individual=individual,protected=protected,
                                   analytic_vectors=len(probes)*len(bp),symbolic=symbolic)))
    witness = dict(r="19/2",x="1",candidate=576,
                   conflict_completion=["19/2","55/4","55/4","29/2"],
                   ignoring_admission_completion=["19/2","55/4","55/4","14"],
                   missed_d_delay="1/2")
    return dict(seed=7880,rounds=rounds,cases=cases,witness=witness,
                analytic_vectors=sum(c['analytic_vectors'] for c in cases),
                symbolic_vectors=sum(c['symbolic'][0]['probes'] for c in cases),
                symbolic_regions=sum(c['symbolic'][0]['cells'] for c in cases),
                triples_rejected=sum(not v for c in cases for v in c['triple_fits'].values()),
                max_pair_points=max(v['points'] for c in cases for v in c['pairs'].values()),
                scope='Four all-arrived requests, disjoint missing objects, pairwise coexistence, fixed C/D; no initial transfers or reservations.',
                theoretical_pair_scenarios=438)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--check',type=Path)
    args=parser.parse_args()
    result=run()
    if args.check:
        assert result==read(args.check)
    if args.out:
        with args.out.open('x',encoding='utf-8') as f:
            json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False))
