"""Certify and execute actual first-entry rollout under one fixed baseline.

All arrivals zero, stationary unit lane speeds, exact active-phase forecasts.
Private score adapter changes only candidate ranking; original selection,
admission, transfer and completion processing execute in Experiment._run.
"""

import argparse
from collections import Counter
from copy import deepcopy
import csv
from fractions import Fraction as F
import json
from pathlib import Path

from scripts.cpu.lookahead.robustness.study_lookahead_composition import Batch
from scripts.cpu.lookahead.screening.study_lookahead_phase1 import Experiment, ROOT, read
from scripts.cpu.lookahead.screening.study_lookahead_workload_bounds import clone, random_cases


class SnapshotBatch(Batch):
    def __init__(self,exp,state,speeds):
        self.exp=exp
        self.state=clone(state)
        self.estimate={}
        self.speeds=speeds
        self.lo,self.hi=F(3,4),F(5,4)
        self.cache={}
        self.bounds={}
        self.points=0


class EntryRollout(Experiment):
    def audit(self,epsilon):
        self.epsilon=epsilon
        self.records=[]
        self.current=None
        self.baseline=Experiment._run(self,'local-cost',0,None,100,len(self.requests))
        self.baseplan=tuple(self.baseline['boundaries_tokens'])
        expected=iter(i for i,c in enumerate(self.eligible) if len(c)>1)

        def select(rows,fallback,later):
            self.target=next(expected)
            assert later==len(self.requests)-self.target-1
            assert fallback==self.baseplan[self.target]
            self.current=None
            return 0

        actual=self.run('snapshot-screened',estimate={},screen_mode='classified',
                        screen_selector=select,score_candidate=lambda row:0,
                        screen_search='lean')
        assert next(expected,None) is None
        ends=actual['completion_ms']
        ledger=[F(0)]*len(ends)
        realized=[F(0)]*len(ends)
        stats=Counter()
        previous=self.baseline['completion_ms']
        log=[]
        for record in self.records:
            head=record['head']
            ref=ends[:head]+record['reference']
            chosen=ends[:head]+record['chosen']
            assert ref==previous,('reference bridge',head,record['target'],ref,previous)
            previous=chosen
            for j,(a,b,u) in enumerate(zip(chosen[head:],ref[head:],record['upper']),head):
                realized[j]+=a-b
                ledger[j]+=u
            assert actual['boundaries_tokens'][record['target']]==record['selected']
            stats['decisions']+=1
            stats['edits']+=record['selected']!=record['fallback']
            stats['running_decisions']+=record['running']
            stats['future_target_decisions']+=record['target']>head
            stats['symbolic_plans']+=record['plans']
            stats['numeric_plan_points']+=record['points']
            log.append({k:record[k] for k in ('head','target','fallback','selected','running')}
                       |{k:list(map(str,record[k])) for k in ('reference','chosen','upper')})
        assert previous==ends
        delta=[a-b for a,b in zip(ends,self.baseline['completion_ms'])]
        assert delta==realized
        assert all(d<=u for d,u in zip(delta,ledger))
        assert max(delta)<=stats['decisions']*epsilon
        assert sum(delta)<=0
        return dict(epsilon=str(epsilon),baseline=json.dumps(self.baseplan),
                    actual_plan=json.dumps(actual['boundaries_tokens']),
                    baseline_completion=json.dumps(list(map(str,self.baseline['completion_ms']))),
                    actual_completion=json.dumps(list(map(str,ends))),
                    ledger=json.dumps(list(map(str,ledger))),decisions_log=json.dumps(log),
                    worst_individual=str(max(delta)),total_delta=str(sum(delta)),**stats)

    def _run(self,policy,threshold_tokens,boundaries,max_combinations,n,**kwargs):
        state=kwargs.get('snapshot')
        if state is None:
            return Experiment._run(self,policy,threshold_tokens,boundaries,max_combinations,n,**kwargs)
        assert policy=='fixed' and n==len(self.requests)==state[2]
        assert self.target not in state[3]
        if self.current is None:
            batch=SnapshotBatch(self,state,kwargs['forecast_speed'])
            base=tuple(self.baseplan[j] if j>=self.target else boundaries[j] for j in range(n))
            # The simulator's frozen suffix must implement this same baseline.
            assert tuple(boundaries[self.target+1:])==base[self.target+1:]
            best=batch.edit(base,self.target,self.epsilon)
            upper,score=batch.difference(best,base)
            assert max(upper)<=self.epsilon and score<=0
            def oracle(plan):
                return Experiment._run(self,'fixed',0,plan,100,n,snapshot=clone(state),
                                       estimate={},forecast_speed=kwargs['forecast_speed'],
                                       lean_rollout=True)['completion_ms']
            record=dict(head=state[1],target=self.target,fallback=base[self.target],
                        selected=best[self.target],running=int(state[5] is not None),
                        reference=oracle(base),chosen=oracle(best),upper=upper,
                        plans=len(batch.cache),points=batch.points)
            self.records.append(record)
            self.current=batch,base
        batch,base=self.current
        assert tuple(boundaries[:self.target])==base[:self.target]
        assert tuple(boundaries[self.target+1:])==base[self.target+1:]
        upper,score=batch.difference(tuple(boundaries),base)
        # Unsafe candidates get a strictly positive score; baseline has zero.
        # These are comparison scalars, not reported completion timestamps.
        return dict(completion_ms=[score if max(upper)<=self.epsilon else F(1)],
                    events_processed=0)


def transformed(trace,ssd_scale):
    trace=deepcopy(trace)
    for request in trace['requests']:
        request['arrival_ms']=0
    trace['io_path_speed_epochs']={p:[[0,'1']] for p in ('ram_to_device','ssd_to_device')}
    cost=trace['cost']['ssd_to_device']
    cost['startup_ms']=str(F(cost['startup_ms'])*ssd_scale)
    cost['bytes_per_ms']=str(F(cost['bytes_per_ms'])/ssd_scale)
    return trace


def main(args):
    model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    rows=[]
    stats=Counter()
    for case,(trace,labels,_) in enumerate(random_cases(model,args.seed,args.rounds)):
        for scale in map(F,args.ssd_scales):
            exp=EntryRollout(transformed(trace,scale),model,labels['budget'])
            row=exp.audit(F(args.epsilon))
            rows.append(dict(seed=args.seed,case=case,capacity=labels['budget'],ssd_scale=str(scale),**row))
            stats['paths']+=1
            for key in ('decisions','edits','running_decisions','future_target_decisions',
                        'symbolic_plans','numeric_plan_points'):
                stats[key]+=row.get(key,0)
            stats['positive_individual_paths']+=F(row['worst_individual'])>0
            stats['positive_total_paths']+=F(row['total_delta'])>0
    stats['cases']=args.rounds
    with args.out.open('x',encoding='utf-8',newline='') as f:
        keys=list(dict.fromkeys(k for row in rows for k in row))
        writer=csv.DictWriter(f,fieldnames=keys,restval=0)
        writer.writeheader()
        writer.writerows(rows)
    with args.summary.open('x',encoding='utf-8') as f:
        json.dump(dict(stats),f,indent=2,sort_keys=True)
    print(json.dumps(dict(stats),sort_keys=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=7750)
    p.add_argument('--rounds',type=int,default=100)
    p.add_argument('--ssd-scales',nargs='+',default=['1/2','1','2'])
    p.add_argument('--epsilon',default='0')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--summary',type=Path,required=True)
    args=p.parse_args()
    assert args.rounds>0 and F(args.epsilon)>=0 and all(F(x)>0 for x in args.ssd_scales)
    main(args)
