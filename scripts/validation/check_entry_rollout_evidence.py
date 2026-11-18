"""Reconstruct first-entry potential chains, budgets and reference bridges."""

import argparse
from collections import Counter
import csv
from fractions import Fraction as F
import json
from pathlib import Path

from scripts.validation.lookahead_evidence import csv_path, csv_paths, summary_path
from scripts.cpu.lookahead.screening.study_lookahead_phase1 import ROOT, read
from scripts.cpu.lookahead.robustness.study_lookahead_entry_rollout import EntryRollout, transformed
from scripts.cpu.lookahead.robustness.study_lookahead_reference_drift import DriftEntry
from scripts.cpu.lookahead.screening.study_lookahead_workload_bounds import random_cases


def check(path,drift=False):
    stats=Counter()
    derived=Counter()
    cases=set()
    for row in csv.DictReader(path.open(encoding='utf-8')):
        base=list(map(F,json.loads(row['baseline_completion'])))
        actual=list(map(F,json.loads(row['actual_completion'])))
        logs=json.loads(row['decisions_log'])
        previous=base
        actions=[F(0)]*len(base)
        bridges=[F(0)]*len(base)
        ledger=[F(0)]*len(base)
        for r in logs:
            head=r['head']
            ref=list(map(F,r['reference']))
            chosen=list(map(F,r['chosen']))
            if not drift:
                ref=actual[:head]+ref
                chosen=actual[:head]+chosen
            bridge=[a-b for a,b in zip(ref,previous)]
            action=[a-b for a,b in zip(chosen,ref)]
            if drift:
                assert bridge==list(map(F,r['bridge']))
                assert action==list(map(F,r['action']))
                assert max(action)<=0
            else:
                assert not any(bridge)
                upper=list(map(F,r['upper']))
                assert len(upper)==len(base)-head
                assert max(upper)<=F(row['epsilon'])
                assert sum(action)<=0
                for j,(d,u) in enumerate(zip(action[head:],upper),head):
                    assert d<=u
                    ledger[j]+=u
            actions=[a+b for a,b in zip(actions,action)]
            bridges=[a+b for a,b in zip(bridges,bridge)]
            previous=chosen
        assert previous==actual
        delta=[a-b for a,b in zip(actual,base)]
        assert delta==[a+b for a,b in zip(actions,bridges)]
        assert F(row['worst_individual'])==max(delta)
        assert F(row['total_delta'])==sum(delta)
        cases.add(row['case'])
        if drift:
            assert actions==list(map(F,json.loads(row['action_sum'])))
            assert bridges==list(map(F,json.loads(row['bridge_sum'])))
            upper=list(map(F,json.loads(row['certified_upper'])))
            assert max(upper)<=0
            assert all(d<=u+b for d,u,b in zip(delta,upper,bridges))
            derived['positive_bridge_cases']+=any(b>0 for b in bridges)
            derived['uncorrected_ledger_violations']+=any(d>u for d,u in zip(delta,upper))
            stats['cases']+=1
            stats['decisions']+=len(logs)
            stats['nonzero_bridge_cases']+=any(bridges)
            stats['positive_individual_cases']+=max(delta)>0
            stats['positive_total_cases']+=sum(delta)>0
        else:
            assert ledger==list(map(F,json.loads(row['ledger'])))
            assert all(d<=u for d,u in zip(delta,ledger))
            assert int(row['decisions'])==len(logs)==3
            assert int(row['edits'])==sum(r['selected']!=r['fallback'] for r in logs)
            assert int(row['running_decisions'])==sum(r['running'] for r in logs)
            assert int(row['future_target_decisions'])==sum(r['target']>r['head'] for r in logs)
            stats['paths']+=1
            for key in ('decisions','edits','running_decisions','future_target_decisions',
                        'symbolic_plans','numeric_plan_points'):
                stats[key]+=int(row[key])
            stats['positive_individual_paths']+=max(delta)>0
            stats['positive_total_paths']+=sum(delta)>0
    if not drift:
        stats['cases']=len(cases)
    assert dict(stats)==json.loads(summary_path(path).read_text()),(path,stats)
    print(path.name,dict(stats))
    if drift:
        print('Derived bridge diagnostics:',dict(derived))


def replay(seed,rounds,epsilon):
    model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    suffix='-cap1' if epsilon else ''
    path=csv_path(f'dynamic-lookahead-entry-rollout-{seed}{suffix}.csv')
    rows={(int(r['case']),r['ssd_scale']):r for r in csv.DictReader(path.open(encoding='utf-8'))}
    count=0
    for case,(trace,labels,_) in enumerate(random_cases(model,seed,rounds)):
        for scale in map(F,('1/2','1','2')):
            row=EntryRollout(transformed(trace,scale),model,labels['budget']).audit(epsilon)
            saved=rows[case,str(scale)]
            for key,value in row.items():
                assert str(value)==saved[key],(case,scale,key)
            count+=1
    print('Replayed first-entry paths:',count)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay-seed',type=int)
    p.add_argument('--rounds',type=int,default=20)
    p.add_argument('--epsilon',default='0')
    p.add_argument('--witness',type=Path)
    args=p.parse_args()
    if args.witness:
        payload=json.loads(args.witness.read_text())
        model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
        result=DriftEntry(payload['trace'],model,payload['capacity']).audit_drift()
        assert all(payload[k]==v for k,v in result.items())
        print('KV reference drift witness replayed:',result['worst_individual'],result['total_delta'])
    elif args.replay_seed is not None:
        assert F(args.epsilon) in (0,1)
        replay(args.replay_seed,args.rounds,F(args.epsilon))
    else:
        for path in csv_paths('dynamic-lookahead-entry-rollout-*.csv'):
            check(path)
        for path in csv_paths('dynamic-lookahead-reference-drift-*.csv'):
            check(path,True)
