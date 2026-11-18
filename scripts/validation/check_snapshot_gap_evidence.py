"""Reconstruct paired model-gap identities and replay fixed-reference paths."""

import argparse
from collections import Counter
import csv
from fractions import Fraction as F
import json
from pathlib import Path
from copy import deepcopy

from scripts.validation.lookahead_evidence import csv_path, csv_paths, summary_path
from scripts.cpu.lookahead.screening.study_lookahead_phase1 import ROOT, read
from scripts.cpu.lookahead.robustness.study_lookahead_snapshot_gap import SnapshotGap, PHASES
from scripts.cpu.lookahead.screening.study_lookahead_workload_bounds import random_cases


def check(path):
    stats={}
    for row in csv.DictReader(path.open(encoding='utf-8')):
        phase=row['phase']
        logs=json.loads(row['decisions_log'])
        base=list(map(F,json.loads(row['baseline_completion'])))
        actual=list(map(F,json.loads(row['actual_completion'])))
        assert len(logs)==3 and len(base)==len(actual)==4
        previous=base
        ledger=[F(0)]*len(base)
        predicted=[F(0)]*len(base)
        gaps=[F(0)]*len(base)
        for r in logs:
            ref=list(map(F,r['reference']))
            chosen=list(map(F,r['chosen']))
            action=list(map(F,r['true_action']))
            pred=list(map(F,r['predicted_action']))
            gap=list(map(F,r['paired_model_gap']))
            upper=list(map(F,r['upper']))
            assert ref==previous
            assert action==[a-b for a,b in zip(chosen,ref)]
            assert gap==[a-b for a,b in zip(action,pred)]
            assert max(upper)<=0 and max(pred)<=0
            assert all(d<=u for d,u in zip(pred,upper))
            if not F(r['running_error']) and not F(r['transfer_error']):
                assert not any(gap)
            previous=chosen
            ledger=[a+b for a,b in zip(ledger,upper)]
            predicted=[a+b for a,b in zip(predicted,pred)]
            gaps=[a+b for a,b in zip(gaps,gap)]
        assert previous==actual
        delta=[a-b for a,b in zip(actual,base)]
        assert delta==[a+b for a,b in zip(predicted,gaps)]
        assert all(d<=u+e for d,u,e in zip(delta,ledger,gaps))
        for key,values in [('ledger',ledger),('predicted_sum',predicted),('gap_sum',gaps)]:
            assert values==list(map(F,json.loads(row[key])))
        assert F(row['worst_individual'])==max(delta)
        assert F(row['total_delta'])==sum(delta)
        assert F(row['uncorrected_ledger_gap'])==max(d-u for d,u in zip(delta,ledger))
        if phase=='exact':
            assert not any(gaps) and max(delta)<=0 and sum(delta)<=0
        st=stats.setdefault(phase,Counter())
        st['cases']+=1
        st['decisions']+=len(logs)
        st['edits']+=sum(a!=b for a,b in zip(json.loads(row['baseline']),json.loads(row['actual_plan'])))
        st['native_reference_mismatch_decisions']+=sum(r['native_fallback']!=r['baseline'] for r in logs)
        st['snapshot_error_decisions']+=sum(F(r['running_error'])!=0 or F(r['transfer_error'])!=0 for r in logs)
        st['positive_individual_cases']+=max(delta)>0
        st['positive_total_cases']+=sum(delta)>0
        st['uncorrected_ledger_violations']+=any(d>u for d,u in zip(delta,ledger))
        for key in ('symbolic_plans','numeric_plan_points'):
            st[key]+=int(row[key])
    assert stats==json.loads(summary_path(path).read_text())
    print(path.name,json.dumps(stats,sort_keys=True))


def replay(seed,rounds):
    path=csv_path(f'dynamic-lookahead-snapshot-gap-{seed}.csv')
    saved={(int(r['case']),r['phase']):r for r in csv.DictReader(path.open(encoding='utf-8'))}
    model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    count=0
    for case,(trace,labels,_) in enumerate(random_cases(model,seed,rounds)):
        trace=deepcopy(trace)
        for r in trace['requests']:
            r['arrival_ms']=0
        for phase in PHASES:
            result=SnapshotGap(trace,model,labels['budget']).audit(phase)
            row=saved[case,phase]
            for key,v in result.items():
                value=json.dumps(v) if isinstance(v,(list,dict)) else str(v)
                assert value==row[key],(seed,case,phase,key)
            count+=1
    print('Replayed paths:',count)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay-seed',type=int)
    p.add_argument('--rounds',type=int,default=20)
    p.add_argument('--witness',type=Path)
    args=p.parse_args()
    if args.witness:
        payload=json.loads(args.witness.read_text())
        model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
        result=SnapshotGap(payload['trace'],model,payload['capacity']).audit(payload['phase'])
        assert all(payload[k]==v for k,v in result.items())
        assert F(result['worst_individual'])>0
        if 'control' in payload:
            control=SnapshotGap(payload['trace'],model,payload['capacity']).audit('exact')
            assert control==payload['control']
            assert control['baseline_completion']==control['actual_completion']
        print('Replayed witness:',args.witness.name,result['worst_individual'],result['total_delta'])
    elif args.replay_seed is not None:
        replay(args.replay_seed,args.rounds)
    else:
        for path in csv_paths('dynamic-lookahead-snapshot-gap-*.csv'):
            check(path)
