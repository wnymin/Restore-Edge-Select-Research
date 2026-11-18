"""Audit state-consistent baseline rollout at actual FIFO handoff states.

Known finite saturated batch; this is not original first-entry window selection.
An admitted or running head is irrevocable. Future unchosen plans follow a
frozen legal baseline. Certificates cover all remaining request completions.
"""

import argparse
import csv
import json
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
from random import Random

from scripts.cpu.lookahead.robustness.study_lookahead_composition import Batch
from scripts.cpu.lookahead.screening.study_lookahead_phase1 import Experiment, ROOT, read
from scripts.cpu.lookahead.screening.study_lookahead_workload_bounds import clone, random_cases


def run_case(trace, model, capacity, estimate, baseline, theta, epsilon):
    batch=Batch(trace,model,capacity,estimate,F(1,4))
    n=len(baseline)
    factors=estimate|{'ssd_to_device':estimate['ssd_to_device']*theta}

    def execute(state,plan,pause=None):
        return Experiment._run(batch.exp,'fixed',0,plan,100,n,snapshot=clone(state),
                               estimate=factors,forecast_speed=batch.speeds,
                               lean_rollout=True,pause_at_head=pause)

    original=execute(batch.state,baseline)['completion_ms']
    plan=baseline
    actual=[]
    ledger=[F(0)]*n
    realized=[F(0)]*n
    stats=Counter()
    for head in range(n):
        state=batch.state
        assert state[1]==head and state[5] is None
        reference=execute(state,plan)['completion_ms']
        assert len(reference)==n-head
        old=plan
        # Changing a previously admitted head would change committed loading
        # semantics; freeze it instead of pretending the certificate allows it.
        if head in state[3]:
            stats['locked_heads']+=1
        elif len(batch.exp.eligible[head])>1:
            stats['certified_decisions']+=1
            plan=batch.edit(old,head,epsilon)
            upper,score=batch.difference(plan,old)
            assert len(upper)==n-head and max(upper)<=epsilon and score<=0
            for j,value in enumerate(upper,head):
                ledger[j]+=value
            stats['edits']+=plan!=old
        candidate=execute(state,plan)['completion_ms']
        for j,(a,b) in enumerate(zip(candidate,reference),head):
            realized[j]+=a-b
        step=execute(state,plan,head+1)
        assert len(step['completion_ms'])==1
        actual.extend(step['completion_ms'])
        stats['symbolic_plans']+=len(batch.cache)
        if head+1<n:
            batch.state=step['snapshot']
            # The base continuation at the next actual state must reproduce
            # exactly the tail of the previous action-plus-base rollout.
            tail=execute(batch.state,plan)['completion_ms']
            assert step['completion_ms']+tail==candidate
            stats['potential_links']+=1
        else:
            assert step['completion_ms']==candidate
        batch.cache.clear()
        batch.bounds.clear()
    delta=[a-b for a,b in zip(actual,original)]
    assert delta==realized
    assert all(d<=u for d,u in zip(delta,ledger))
    assert max(delta)<=stats['certified_decisions']*epsilon
    assert sum(delta)<=0
    if epsilon==0:
        assert max(delta)<=0
    return dict(theta=str(theta),epsilon=str(epsilon),
                baseline=json.dumps(baseline),final_plan=json.dumps(plan),
                baseline_completion=json.dumps(list(map(str,original))),
                actual_completion=json.dumps(list(map(str,actual))),
                ledger=json.dumps(list(map(str,ledger))),
                realized_differences=json.dumps(list(map(str,realized))),
                worst_individual=str(max(delta)),total_delta=str(sum(delta)),
                strict_individual_improvement=int(any(d<0 for d in delta)),
                certified_decisions=stats['certified_decisions'],edits=stats['edits'],
                locked_heads=stats['locked_heads'],potential_links=stats['potential_links'],
                symbolic_plans=stats['symbolic_plans'],numeric_plan_points=batch.points)


def main(args):
    model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    rng=Random(args.seed+1)
    rows=[]
    stats=Counter()
    for case,(trace,labels,estimate) in enumerate(random_cases(model,args.seed,args.rounds)):
        exp=Experiment(trace,model,labels['budget'])
        baseline=tuple(rng.choice(choices) for choices in exp.eligible)
        for theta in map(F,args.theta):
            row=run_case(trace,model,labels['budget'],estimate,baseline,theta,F(args.epsilon))
            rows.append(dict(seed=args.seed,case=case,capacity=labels['budget'],**row))
            stats['paths']+=1
            for key in ('certified_decisions','edits','locked_heads','potential_links',
                        'symbolic_plans','numeric_plan_points','strict_individual_improvement'):
                stats[key]+=row[key]
            stats['positive_individual_paths']+=F(row['worst_individual'])>0
            stats['positive_total_paths']+=F(row['total_delta'])>0
    stats['cases']=args.rounds
    with args.out.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with args.summary.open('x',encoding='utf-8') as f:
        json.dump(dict(stats),f,indent=2,sort_keys=True)
    print(json.dumps(dict(stats),sort_keys=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=7740)
    p.add_argument('--rounds',type=int,default=100)
    p.add_argument('--theta',nargs='+',default=['3/4','7/8','1','9/8','5/4'])
    p.add_argument('--epsilon',default='0')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--summary',type=Path,required=True)
    args=p.parse_args()
    assert args.rounds>0 and F(args.epsilon)>=0
    assert all(F(3,4)<=F(t)<=F(5,4) for t in args.theta)
    main(args)
