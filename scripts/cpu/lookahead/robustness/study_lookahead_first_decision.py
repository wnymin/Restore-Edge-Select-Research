"""Audit shared-residual minimax regret and individual caps at first decisions."""

import argparse
import csv
import json
from collections import Counter
from fractions import Fraction as F

from scripts.cpu.lookahead.screening.study_lookahead_trigger import Trigger, cases
from scripts.cpu.lookahead.screening.study_lookahead_phase1 import Experiment, ROOT, read
from scripts.cpu.lookahead.screening.study_lookahead_workload_bounds import clone, random_cases
from scripts.cpu.lookahead.robustness.study_lookahead_residual import Affine, Domain, partition
from scripts.cpu.lookahead.robustness.study_lookahead_candidate_difference import extrema


class Capture(Trigger):
    def __init__(self, *args):
        super().__init__(*args)
        self.records=[]

    def _run(self, policy, threshold_tokens, boundaries, max_combinations, n, **kwargs):
        state=kwargs.get('snapshot')
        if state is not None and (not self.records or self.records[-1][0]!=self.i):
            assert self.i not in state[3]
            self.records.append((self.i,self.fallback,clone(state),tuple(boundaries),
                                 dict(kwargs['estimate']),dict(kwargs['forecast_speed'])))
        return Experiment._run(self,policy,threshold_tokens,boundaries,max_combinations,n,**kwargs)


def vector_at(cells,x):
    return next(ends for d,ends,_ in cells if d.contains(x))


def analyze(exp,record,radius,cap,parameter='residual'):
    target,fallback,state,base,estimate,speeds=record
    residual=state[6][0]-state[0] if parameter=='residual' else F(1)
    lo,hi=residual*(1-radius),residual*(1+radius)
    plans={c:tuple(c if j==target else b for j,b in enumerate(base)) for c in exp.eligible[target]}
    tables={c:partition(exp,state,p,estimate,speeds,lo,hi,parameter) for c,p in plans.items()}
    points=sorted({x for cells in tables.values() for d,_,_ in cells for x in (d.lo,d.hi)})
    domains=[Domain(x,x) for x in points]
    domains += [Domain(a,b,False,False) for a,b in zip(points,points[1:]) if a<b]
    domains.sort(key=lambda d:(d.lo,d.hi))
    pairs={(a,b):[] for a in plans for b in plans}
    individual={c:[] for c in plans}
    probes=set(points)|{residual}
    for d in domains:
        x=(d.lo+d.hi)/2
        vectors={c:vector_at(t,x) for c,t in tables.items()}
        totals={c:sum(v,Affine()) for c,v in vectors.items()}
        for a,b in pairs:
            delta=totals[a]-totals[b]
            pairs[a,b].append((d,delta))
            if delta.b and d.contains(-delta.a/delta.b):
                probes.add(-delta.a/delta.b)
        for c in plans:
            for a,b in zip(vectors[c],vectors[fallback]):
                delta=a-b
                individual[c].append((d,delta))
                # A cap crossing is also checked against the numeric oracle.
                if delta.b and d.contains((cap-delta.a)/delta.b):
                    probes.add((cap-delta.a)/delta.b)
        if d.lo<d.hi:
            probes.update(((3*d.lo+d.hi)/4,x,(d.lo+3*d.hi)/4))
    uppers={key:extrema(cells)[1] for key,cells in pairs.items()}
    regrets={a:max(uppers[a,b] for b in plans) for a in plans}
    caps={c:extrema(v)[1] for c,v in individual.items()}
    safe=[c for c in plans if caps[c]<=cap]
    assert fallback in safe
    # Prefer fallback on equal regret, then deeper boundary: explicit new rule.
    choose=lambda allowed:min(allowed,key=lambda c:(regrets[c],c!=fallback,-c))
    robust=choose(plans)
    constrained=choose(safe)
    exact=Counter()
    numeric_regrets={c:F(0) for c in plans}
    numeric_caps={c:F(0) for c in plans}
    nominal=None
    for x in sorted(probes):
        changed=(*state[:6],(state[0]+x,state[6][1]),*state[7:]) if parameter=='residual' else state
        costs=estimate if parameter=='residual' else estimate|{'ssd_to_device':estimate.get('ssd_to_device',1)*x}
        vectors={}
        for c,plan in plans.items():
            out=Experiment._run(exp,'fixed',0,plan,100,state[2],snapshot=clone(changed),
                                estimate=costs,forecast_speed=speeds,lean_rollout=True)
            vectors[c]=out['completion_ms']
            assert vectors[c]==[a.at(x) for a in vector_at(tables[c],x)]
        totals={c:sum(v) for c,v in vectors.items()}
        best=min(totals,key=lambda c:(totals[c],-c))
        if totals[best]>=totals[fallback]:
            best=fallback
        exact[best]+=1
        if x==residual:
            nominal=best
        for c in plans:
            numeric_regrets[c]=max(numeric_regrets[c],totals[c]-min(totals.values()))
            numeric_caps[c]=max(numeric_caps[c],max(a-b for a,b in zip(vectors[c],vectors[fallback])))
    for c in plans:
        assert numeric_regrets[c]<=regrets[c]
        assert numeric_caps[c]<=max(F(0),caps[c])
    # Independent vector evaluation at each open-branch limit reconstructs the
    # envelope regret; limit points are not passed as actual endpoint schedules.
    envelope={c:F(0) for c in plans}
    for d in domains:
        vectors={c:vector_at(t,(d.lo+d.hi)/2) for c,t in tables.items()}
        for x in (d.lo,d.hi):
            costs={c:sum(a.at(x) for a in v) for c,v in vectors.items()}
            for c in plans:
                envelope[c]=max(envelope[c],costs[c]-min(costs.values()))
    assert envelope==regrets
    common=dict(target=target,head=state[1],running=int(state[5] is not None),
                distance=target-state[1],candidates=len(plans),probes=len(probes),cells=len(domains),
                fallback=fallback,nominal=nominal,robust=robust,constrained=constrained,
                winners=len(exact),radius=str(radius),cap=str(cap),residual=str(residual),
                parameter=parameter,inflight=int(state[6] is not None),
                parameter_sensitive=int(any(a.b for cells in tables.values() for _,v,_ in cells for a in v)))
    rows=[common|dict(candidate=c,regret=str(regrets[c]),individual_upper=str(caps[c]),
                      safe=int(c in safe),selected=int(c==constrained)) for c in plans]
    return rows


def main(args):
    model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    source=(random_cases(model,args.seed,args.rounds) if args.split=='random' else
            ((t,l,{'ssd_to_device':F(l['estimate'])}) for t,l in cases(args,model)))
    stats=Counter()
    rows=[]
    for case,(trace,labels,estimate) in enumerate(source):
        if stats['decisions']>=args.limit:
            break
        exp=Capture(trace,model,labels['budget'])
        exp.run_trigger(estimate,'full')
        stats['configurations']+=1
        stats['captured_first_decisions']+=len(exp.records)
        for record in exp.records:
            if stats['decisions']>=args.limit:
                break
            state=record[2]
            if args.parameter=='residual' and (state[6] is None or state[6][0]<=state[0]):
                stats['skipped_no_positive_inflight']+=1
                continue
            result=analyze(exp,record,F(args.radius),F(args.cap),args.parameter)
            stats['decisions']+=1
            first=result[0]
            stats['candidate_actions']+=len(result)
            stats['probes']+=first['probes']
            stats['cells']+=first['cells']
            stats['running_decisions']+=first['running']
            stats['no_inflight_decisions']+=not first['inflight']
            stats['parameter_sensitive_decisions']+=first['parameter_sensitive']
            stats['future_target_decisions']+=first['distance']>0
            stats['three_plus_candidates']+=len(result)>=3
            stats['winner_changes']+=first['winners']>1
            stats['robust_differs_nominal']+=first['robust']!=first['nominal']
            stats['constraint_changes_action']+=first['robust']!=first['constrained']
            stats['constrained_differs_fallback']+=first['constrained']!=first['fallback']
            stats['unsafe_actions']+=sum(not row['safe'] for row in result)
            rows.extend(dict(seed=args.seed,case=case,decision=stats['decisions'],**row) for row in result)
    with args.out.open('x',encoding='utf-8',newline='') as f:
        fields=tuple(rows[0]) if rows else ('seed','case','decision','candidate','regret','individual_upper','safe','selected')
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with args.summary.open('x',encoding='utf-8') as f:
        json.dump(dict(stats),f,indent=2,sort_keys=True)
    print(json.dumps(dict(stats),sort_keys=True))


if __name__=='__main__':
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=7710)
    parser.add_argument('--rounds',type=int,default=500)
    parser.add_argument('--limit',type=int,default=500)
    parser.add_argument('--split',choices=('random','witness'),default='random')
    parser.add_argument('--radius',default='1/4')
    parser.add_argument('--cap',default='0')
    parser.add_argument('--parameter',choices=('residual','ssd-scale'),default='residual')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--summary',type=Path,required=True)
    args=parser.parse_args()
    assert 0<F(args.radius)<1 and F(args.cap)>=0
    main(args)
