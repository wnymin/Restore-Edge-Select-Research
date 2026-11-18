"""Reconstruct rollout ledgers and the missing-reference-bridge counterexample."""

import argparse
import csv
import json
from collections import Counter
from fractions import Fraction as F
from pathlib import Path

from scripts.validation.lookahead_evidence import csv_path, csv_paths, summary_path
from scripts.cpu.lookahead.screening.study_lookahead_phase1 import ROOT, Experiment, read
from scripts.cpu.lookahead.screening.study_lookahead_workload_bounds import random_cases
from scripts.cpu.lookahead.robustness.study_lookahead_causal_rollout import run_case
from random import Random


def check(path):
    stats=Counter()
    cases=set()
    for row in csv.DictReader(path.open(encoding='utf-8')):
        original=list(map(F,json.loads(row['baseline_completion'])))
        actual=list(map(F,json.loads(row['actual_completion'])))
        ledger=list(map(F,json.loads(row['ledger'])))
        realized=list(map(F,json.loads(row['realized_differences'])))
        assert len(original)==len(actual)==len(ledger)==len(realized)==4
        delta=[a-b for a,b in zip(actual,original)]
        assert delta==realized
        assert all(d<=u for d,u in zip(delta,ledger))
        assert max(delta)==F(row['worst_individual'])
        assert sum(delta)==F(row['total_delta'])<=0
        assert max(delta)<=int(row['certified_decisions'])*F(row['epsilon'])
        assert int(row['potential_links'])==3
        assert int(row['strict_individual_improvement'])==any(d<0 for d in delta)
        cases.add(row['case'])
        stats['paths']+=1
        for key in ('certified_decisions','edits','locked_heads','potential_links',
                    'symbolic_plans','numeric_plan_points','strict_individual_improvement'):
            stats[key]+=int(row[key])
        stats['positive_individual_paths']+=max(delta)>0
        stats['positive_total_paths']+=sum(delta)>0
    stats['cases']=len(cases)
    archived=json.loads(summary_path(path).read_text())
    assert dict(stats)==archived,(path,stats,archived)
    print(path.name,dict(stats))


def reference_bridge():
    # An abstract two-decision tree, not a KV event instance. First choose a
    # terminal base branch (cost 1) or enter state A. Base continuation at A is
    # z (cost 0); a replacement reference u costs 100, and its proposed v 99.
    terminal={'B':F(1),'z':F(0),'u':F(100),'v':F(99)}
    first_gap=terminal['z']-terminal['B']
    second_proxy_gap=terminal['v']-terminal['u']
    bridge=terminal['u']-terminal['z']
    actual=terminal['v']-terminal['B']
    assert first_gap<=0 and second_proxy_gap<=0 and actual>0
    assert actual==first_gap+bridge+second_proxy_gap
    # Retaining z as reference would reject v, and the safe rollout ends at 0.
    assert terminal['v']-terminal['z']>0
    return {k:str(v) for k,v in dict(first_gap=first_gap,second_proxy_gap=second_proxy_gap,
                                    reference_bridge=bridge,actual_delta=actual).items()}


def replay(seed,rounds,epsilon):
    model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    rng=Random(seed+1)
    suffix = '-cap1' if epsilon else ''
    path=csv_path(f'dynamic-lookahead-causal-rollout-{seed}{suffix}.csv')
    rows={(int(r['case']),r['theta']):r for r in csv.DictReader(path.open(encoding='utf-8'))}
    count=0
    for case,(trace,labels,estimate) in enumerate(random_cases(model,seed,rounds)):
        exp=Experiment(trace,model,labels['budget'])
        baseline=tuple(rng.choice(choices) for choices in exp.eligible)
        for theta in map(F,('3/4','7/8','1','9/8','5/4')):
            row=run_case(trace,model,labels['budget'],estimate,baseline,theta,epsilon)
            saved=rows[case,str(theta)]
            for key,value in row.items():
                assert str(value)==saved[key],(seed,case,theta,key,value,saved[key])
            count+=1
    print('Replayed',count,'actual-state paths; all archived fields identical.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay-seed',type=int)
    p.add_argument('--rounds',type=int,default=20)
    p.add_argument('--epsilon',default='0')
    args=p.parse_args()
    print('Abstract reference drift:',reference_bridge())
    if args.replay_seed is not None:
        replay(args.replay_seed,args.rounds,F(args.epsilon))
    else:
        for path in csv_paths('dynamic-lookahead-causal-rollout-*.csv'):
            check(path)
