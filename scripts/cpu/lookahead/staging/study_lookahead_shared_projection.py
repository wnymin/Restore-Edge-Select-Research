"""Exact three-request projection with shared read-only KV, distinct checkpoints."""

import argparse
from fractions import Fraction as F
import json
from pathlib import Path
from random import Random

from scripts.cpu.lookahead.staging.study_lookahead_three_request import (
    Projection, Profile, audit_trace, critical_points, envelopes, profile, seed_trace,
)
from scripts.cpu.lookahead.robustness.study_lookahead_joint_domain import Capture, Experiment, Linear, ROOT, Region, analyze, oracle, read
from scripts.cpu.lookahead.staging.study_lookahead_two_request import jsonable


class SharedProjection(Projection):
    def __init__(self, exp, record, *, shared_checkpoints=False):
        target, fallback, state, base, estimate, speeds = record
        assert (state[1],target,state[2]) == (0,1,3)
        assert state[5] is not None and state[5] > state[0] and state[6] is None
        assert not state[3] and state[4] == exp.initial
        assert not state[8] and state[9] is None and not state[10]
        assert not exp.per_tier and not exp.dynamic and not exp.host_cache_bytes
        assert exp.release_at == 'start' and exp.window >= 2
        assert max(exp.arrivals) <= state[0]
        self.now,self.fallback = state[0],fallback
        self.follower = profile(exp,2,base[2],estimate,speeds)
        self.targets = {c:profile(exp,1,c,estimate,speeds) for c in exp.eligible[1]}
        assert exp.size(self.follower.missing) <= exp.budget
        self.coexist,self.remaining,self.shared = {},{},{}
        for c,p in self.targets.items():
            assert exp.size(p.missing) <= exp.budget
            assert not (p.missing | self.follower.missing).intersection(state[7][:-1])
            shared = p.missing & self.follower.missing
            kv = frozenset(o for o in shared if exp.objects[o]['kind']=='attention_kv')
            checkpoints = shared-kv
            assert kv.issubset(exp.candidates[1][c][:-1])
            assert shared_checkpoints or not checkpoints
            if checkpoints:
                assert checkpoints == {exp.candidates[1][c][-1]}
                assert checkpoints == {exp.candidates[2][base[2]][-1]}
            self.shared[c] = shared
            self.coexist[c] = exp.size(p.missing | self.follower.missing) <= exp.budget
            if checkpoints and self.coexist[c] and c < exp.lengths[1]:
                obj = next(iter(checkpoints))
                copy = exp.path_time('device_copy',exp.objects[obj]['bytes'])*estimate.get('device_copy',1)
                p = Profile(p.load,p.compute+copy,p.missing)
                self.targets[c] = p
            # Concurrent reservation retains KV; delayed admission needs a
            # positive active target phase to hand KV over before completion.
            reusable = shared if self.coexist[c] else (kv if p.compute > 0 else frozenset())
            saved = Linear()
            for obj in reusable:
                path = {'RAM':'ram_to_device','SSD':'ssd_to_device'}[exp.objects[obj]['tier']]
                t = exp._transfer_times[obj]*estimate.get(path,1)/speeds[path]
                saved += Linear(0,0,t) if path=='ssd_to_device' else Linear(t)
            self.remaining[c] = self.follower.load-saved

    def vector(self,c,point):
        p = self.targets[c]
        load,later = p.load.at(point),self.remaining[c].at(point)
        start = max(point[0],load)
        finish = start+p.compute
        ready = (load if self.coexist[c] else start)+later
        return [self.now+point[0],self.now+finish,
                self.now+max(finish,ready)+self.follower.compute]

    def hinges(self,c):
        p = self.targets[c]
        later = self.remaining[c]
        r = Linear(0,1,0)
        lines = [r-p.load,later-p.compute]
        if self.coexist[c]:
            lines.append(r-p.load-later+p.compute)
        return lines


def shared_trace():
    trace = seed_trace()
    trace['requests'][2]['input_token_runs'][0]['token_id'] = 20
    trace['objects'] = [o for o in trace['objects'] if not o['id'].startswith('C.kv')]
    for obj in trace['objects']:
        if obj['id'].startswith('C.state'):
            obj['prefix_token_runs'][0]['token_id'] = 20
    for candidate in trace['requests'][2]['candidates']:
        candidate['load_order'] = [o.replace('C.kv','B.kv') for o in candidate['load_order']]
    return trace


def run(seed=7800,rounds=96):
    rng = Random(seed)
    model = read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    cases=[]
    for case in range(rounds):
        trace=shared_trace()
        for obj in trace['objects']:
            obj['tier']=rng.choice(('RAM','SSD'))
        trace['scheduling']['initial_device_object_ids']=[o['id'] for o in trace['objects'] if rng.randrange(5)==0]
        trace['cost']['device_copy']=dict(startup_ms='1/10',bytes_per_ms=str(1<<20))
        for path in ('ram_to_device','ssd_to_device'):
            trace['cost'][path]=dict(startup_ms=str(rng.randrange(3)),
                                     bytes_per_ms=str(F(16<<20,rng.randrange(1,17))))
        capacity=rng.choice((96,128,160,192))
        fixed=rng.choice((576,1152))
        exp=Capture(trace,model,capacity)
        exp.run_trigger({},'full')
        first=exp.records[0]
        record=(*first[:3],(*first[3][:2],fixed),*first[4:])
        proj=SharedProjection(exp,record)
        residual=record[2][5]-record[2][0]
        box=Region.rectangle(residual/4,residual*F(7,4),F(1,4),F(7,4))
        exact=envelopes(proj,box)
        symbolic=analyze(exp,record,rectangle=box)
        for row in symbolic:
            c=row['candidate']
            assert F(row['individual_upper'])==exact['individual'][c],(case,c,'individual')
            assert F(row['total_upper'])==exact['total'][c],(case,c,'total')
            assert F(row['regret'])==exact['regret'][c],(case,c,'regret')
            assert row['protected']==exact['protected']
        points=critical_points([d for c in proj.targets for d in proj.hinges(c)],box)
        for point in points:
            for c in proj.targets:
                assert proj.vector(c,point)==oracle(exp,record[2],(0,c,fixed),{},record[5],point),(case,c,point)
        cases.append(jsonable(dict(case=case,capacity=capacity,fixed=fixed,
                                  coexist=proj.coexist,remaining={c:(v.a,v.x) for c,v in proj.remaining.items()},
                                  shared={c:sorted(v) for c,v in proj.shared.items()},exact=exact,
                                  symbolic_cells=symbolic[0]['cells'],symbolic_vectors=symbolic[0]['probes'],
                                  critical_vectors=len(points)*len(proj.targets),
                                  zero_compute_conflicts=sum(not proj.coexist[c] and p.compute==0 and bool(proj.shared[c])
                                                             for c,p in proj.targets.items()))))
    return dict(seed=seed,configurations=rounds,cases=cases,
                symbolic_cells=sum(c['symbolic_cells'] for c in cases),
                symbolic_vectors=sum(c['symbolic_vectors'] for c in cases),
                critical_vectors=sum(c['critical_vectors'] for c in cases),
                zero_compute_conflicts=sum(c['zero_compute_conflicts'] for c in cases))


if __name__=='__main__':
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
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},sort_keys=True))
