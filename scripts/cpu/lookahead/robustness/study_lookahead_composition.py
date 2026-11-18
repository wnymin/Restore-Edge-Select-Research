"""Audit chained versus stale-baseline certificates for full batch plans."""

import argparse
import csv
import json
from collections import Counter
from fractions import Fraction as F
from random import Random
from pathlib import Path

from scripts.cpu.lookahead.screening.study_lookahead_workload_bounds import random_cases, clone
from scripts.cpu.lookahead.screening.study_lookahead_phase1 import Experiment, ROOT, read
from scripts.cpu.lookahead.robustness.study_lookahead_residual import Affine, partition
from scripts.cpu.lookahead.robustness.study_lookahead_first_decision import vector_at


class Batch:
    def __init__(self,trace,model,budget,estimate,radius):
        self.exp=Experiment(trace,model,budget)
        self.estimate=estimate
        self.speeds={p:v[1][0] for p,v in self.exp.io_epochs.items()}
        n=len(self.exp.requests)
        # Deliberately a known saturated batch, not natural online arrivals.
        self.state=(max(self.exp.arrivals),0,n,{},set(self.exp.initial),None,None,(),set(),None,{})
        self.lo,self.hi=1-radius,1+radius
        self.cache={}
        self.bounds={}
        self.points=0

    def table(self,plan):
        if plan not in self.cache:
            cells=partition(self.exp,self.state,plan,self.estimate,self.speeds,
                            self.lo,self.hi,'ssd-scale')
            for d,vector,_ in cells:
                probes={(d.lo+d.hi)/2}
                if d.left:
                    probes.add(d.lo)
                if d.right:
                    probes.add(d.hi)
                for x in probes:
                    factors=self.estimate|{'ssd_to_device':self.estimate.get('ssd_to_device',1)*x}
                    out=Experiment._run(self.exp,'fixed',0,plan,100,len(plan),
                                        snapshot=clone(self.state),estimate=factors,
                                        forecast_speed=self.speeds,lean_rollout=True)
                    assert out['completion_ms']==[a.at(x) for a in vector]
                    self.points+=1
            self.cache[plan]=cells
        return self.cache[plan]

    def difference(self,a,b):
        key=(a,b)
        if key in self.bounds:
            return self.bounds[key]
        left,right=self.table(a),self.table(b)
        thresholds=sorted({x for cells in (left,right) for d,_,_ in cells for x in (d.lo,d.hi)})
        probes=[(x,x,x) for x in thresholds]
        probes += [(lo,hi,(lo+hi)/2) for lo,hi in zip(thresholds,thresholds[1:])]
        upper=[None]*len(a)
        sum_upper=None
        for lo,hi,sample in probes:
            diffs=[x-y for x,y in zip(vector_at(left,sample),vector_at(right,sample))]
            for x in (lo,hi):
                values=[d.at(x) for d in diffs]
                upper=[v if u is None else max(u,v) for u,v in zip(upper,values)]
                score=sum(values)
                sum_upper=score if sum_upper is None else max(sum_upper,score)
        self.bounds[key]=(tuple(upper),sum_upper)
        return self.bounds[key]

    def edit(self,reference,target,budget):
        options=[]
        for c in self.exp.eligible[target]:
            candidate=tuple(c if j==target else b for j,b in enumerate(reference))
            individual,score=self.difference(candidate,reference)
            if max(individual)<=budget:
                options.append((score,c!=reference[target],-c,candidate))
        return min(options)[-1]


def main(args):
    model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    rng=Random(args.seed+1)
    stats=Counter()
    rows=[]
    saved=False
    for case,(trace,labels,estimate) in enumerate(random_cases(model,args.seed,args.rounds)):
        batch=Batch(trace,model,labels['budget'],estimate,F(args.radius))
        exp=batch.exp
        baseline=tuple(rng.choice(choices) for choices in exp.eligible)
        targets=[i for i,choices in enumerate(exp.eligible) if len(choices)>1]
        independent=[batch.edit(baseline,i,F(args.epsilon)) for i in targets]
        combined=list(baseline)
        for i,plan in zip(targets,independent):
            combined[i]=plan[i]
        combined=tuple(combined)
        chained=baseline
        ledger=[F(0)]*len(baseline)
        chain_edits=0
        for i in targets:
            updated=batch.edit(chained,i,F(args.epsilon))
            diff,score=batch.difference(updated,chained)
            assert max(diff)<=F(args.epsilon) and score<=0
            ledger=[l+v for l,v in zip(ledger,diff)]
            chain_edits+=updated!=chained
            chained=updated
        final,chain_score=batch.difference(chained,baseline)
        assert all(d<=bound for d,bound in zip(final,ledger))
        assert max(final)<=len(targets)*F(args.epsilon) and chain_score<=0
        stale,stale_score=batch.difference(combined,baseline)
        single_uppers=[batch.difference(p,baseline)[0] for p in independent]
        stale_ledger=[sum(d[j] for d in single_uppers) for j in range(len(baseline))]
        violation=max(d-b for d,b in zip(stale,stale_ledger))
        independent_edits=sum(p!=baseline for p in independent)
        stats['cases']+=1
        stats['symbolic_plans']+=len(batch.cache)
        stats['numeric_plan_points']+=batch.points
        stats['independent_edits']+=independent_edits
        stats['chain_edits']+=chain_edits
        stats['multiple_independent_edits_cases']+=independent_edits>=2
        stats['stale_ledger_violations']+=violation>0
        stats['stale_budget_violations']+=max(stale)>len(targets)*F(args.epsilon)
        stats['stale_total_worse']+=stale_score>0
        stats['chained_differs_stale']+=chained!=combined
        rows.append(dict(seed=args.seed,case=case,capacity=labels['budget'],radius=args.radius,
                         epsilon=args.epsilon,baseline=str(baseline),independent=str(combined),
                         chained=str(chained),independent_edits=independent_edits,chain_edits=chain_edits,
                         stale_upper=str(max(stale)),chain_upper=str(max(final)),
                         stale_total_upper=str(stale_score),chain_total_upper=str(chain_score),
                         stale_ledger_violation=str(violation),ledger=str(ledger),
                         plans=len(batch.cache),numeric_points=batch.points))
        if args.witness and max(stale)>len(targets)*F(args.epsilon) and not saved:
            payload=dict(trace=trace,capacity=labels['budget'],estimate={k:str(v) for k,v in estimate.items()},
                         radius=args.radius,epsilon=args.epsilon,baseline=baseline,targets=targets,
                         individual_plans=independent,combined=combined,chained=chained,
                         stale_individual_upper=[str(v) for v in stale],
                         isolated_uppers=[[str(v) for v in u] for u in single_uppers],
                         stale_total_upper=str(stale_score))
            with args.witness.open('x',encoding='utf-8') as f:
                json.dump(payload,f,indent=2)
            saved=True
    with args.out.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with args.summary.open('x',encoding='utf-8') as f:
        json.dump(dict(stats),f,indent=2,sort_keys=True)
    print(json.dumps(dict(stats),sort_keys=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=7730)
    p.add_argument('--rounds',type=int,default=100)
    p.add_argument('--radius',default='1/4')
    p.add_argument('--epsilon',default='0')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--witness',type=Path)
    args=p.parse_args()
    assert 0<F(args.radius)<1 and F(args.epsilon)>=0
    main(args)
