"""Reconstruct composition counts and replay a stale-certificate failure."""

import argparse
import ast
import csv
import json
from collections import Counter
from fractions import Fraction as F
from pathlib import Path

from scripts.validation.lookahead_evidence import csv_paths, summary_path
from scripts.cpu.lookahead.robustness.study_lookahead_composition import Batch
from scripts.cpu.lookahead.screening.study_lookahead_phase1 import Experiment, ROOT, read
from scripts.cpu.lookahead.screening.study_lookahead_workload_bounds import clone


def check_counts(path):
    stats=Counter()
    for r in csv.DictReader(path.open(encoding='utf-8')):
        baseline=ast.literal_eval(r['baseline'])
        stale=ast.literal_eval(r['independent'])
        chain=ast.literal_eval(r['chained'])
        assert len(baseline)==len(stale)==len(chain)
        assert int(r['independent_edits'])==sum(a!=b for a,b in zip(baseline,stale))
        assert int(r['chain_edits'])==sum(a!=b for a,b in zip(baseline,chain))
        # Ledgers use Fraction repr and are deliberately not evaluated as code.
        # This archived generator has a single-candidate first request and
        # three multi-candidate successors; don't claim arbitrary plan coverage.
        assert len(baseline)==4
        steps=len(baseline)-1
        assert F(r['chain_upper'])<=steps*F(r['epsilon'])
        assert F(r['chain_total_upper'])<=0
        stats['cases']+=1
        stats['symbolic_plans']+=int(r['plans'])
        stats['numeric_plan_points']+=int(r['numeric_points'])
        stats['independent_edits']+=int(r['independent_edits'])
        stats['chain_edits']+=int(r['chain_edits'])
        stats['multiple_independent_edits_cases']+=int(r['independent_edits'])>=2
        stats['stale_ledger_violations']+=F(r['stale_ledger_violation'])>0
        stats['stale_budget_violations']+=F(r['stale_upper'])>steps*F(r['epsilon'])
        stats['stale_total_worse']+=F(r['stale_total_upper'])>0
        stats['chained_differs_stale']+=chain!=stale
    summary=json.loads(summary_path(path).read_text())
    assert dict(stats)==summary,(path,stats,summary)
    print(path.name,dict(stats))


def replay(path,out):
    payload=json.loads(path.read_text())
    model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    estimate={p:F(v) for p,v in payload['estimate'].items()}
    batch=Batch(payload['trace'],model,payload['capacity'],estimate,F(payload['radius']))
    base=tuple(payload['baseline'])
    combined=tuple(payload['combined'])
    chained=tuple(payload['chained'])
    isolated=[tuple(p) for p in payload['individual_plans']]
    for plan in isolated:
        assert max(batch.difference(plan,base)[0])<=F(payload['epsilon'])
    upper,_=batch.difference(combined,base)
    assert [str(v) for v in upper]==payload['stale_individual_upper']
    plans={'baseline':base,'combined':combined,'chained':chained}
    plans.update({f'isolated_{i}':p for i,p in enumerate(isolated)})
    tables=[batch.table(plan) for plan in plans.values()]
    points={F(1)}
    for cells in tables:
        for domain,_,_ in cells:
            points.add((domain.lo+domain.hi)/2)
            if domain.left:
                points.add(domain.lo)
            if domain.right:
                points.add(domain.hi)
    worst=None
    for x in sorted(points):
        vectors={}
        costs=estimate|{'ssd_to_device':estimate.get('ssd_to_device',1)*x}
        for name,plan in plans.items():
            result=Experiment._run(batch.exp,'fixed',0,plan,100,len(plan),snapshot=clone(batch.state),
                                   estimate=costs,forecast_speed=batch.speeds,lean_rollout=True)
            vectors[name]=result['completion_ms']
        violation=max(a-b for a,b in zip(vectors['combined'],vectors['baseline']))
        if worst is None or violation>worst[0]:
            worst=(violation,x,vectors)
    violation,x,vectors=worst
    assert violation>len(payload['targets'])*F(payload['epsilon'])
    for name,values in vectors.items():
        if name.startswith('isolated_'):
            assert all(a-b<=F(payload['epsilon']) for a,b in zip(values,vectors['baseline']))
    if out is not None:
        with out.open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=('plan','boundaries','ssd_multiplier','request','completion','delta_from_baseline'))
            writer.writeheader()
            for name,values in vectors.items():
                for i,(a,b) in enumerate(zip(values,vectors['baseline'])):
                    writer.writerow(dict(plan=name,boundaries=str(plans[name]),ssd_multiplier=str(x),
                                         request=i,completion=str(a),delta_from_baseline=str(a-b)))
    print('Replayed witness:',path.name,'parameter',x,'individual harm',violation)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--witness',type=Path)
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.witness:
        replay(args.witness,args.out)
    else:
        for path in csv_paths('dynamic-lookahead-composition-*.csv'):
            check_counts(path)
