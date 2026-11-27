"""R08-CPU-04: run a single candidate and its ablations on published inputs."""
import argparse
from collections import Counter
from dataclasses import replace
from fractions import Fraction as F
import gzip
from itertools import permutations
import json
from math import ceil
from pathlib import Path
import platform
from statistics import median

from scripts.cpu.replay import study_r08_native as native
from scripts.cpu.replay import study_r08_replacement as replacement
from scripts.cpu.replay.study_r08_counterexamples import (
    B, EVIDENCE, arrivals, capacity, extension, filter_r08, preflight, records,
)
from restore_edge_select.entry_refinement import EntryRefiner, Refinement

ROOT = capacity.ROOT
CFG = ROOT/'configs/r08-entry-refinement.json'
BASE = 'entry-replay'
LOCAL = 'local-cost'
CANDIDATE = 'entry-refinement'
POLICIES = (LOCAL, BASE, CANDIDATE)
ABLATIONS = {
    'short-only': dict(wide=False, joint=False),
    'without-short': dict(short=False),
    'without-joint': dict(joint=False),
    'without-guard': dict(guard=False),
}


def capacity_tasks(cfg, split, rows, keep=lambda k,r: True):
    for key,data,model,scale in capacity.cases(cfg,split):
        row = rows[tuple(key.items())]
        if keep(key,row):
            yield row, capacity.open_exp(data,model,key['rate'],key['capacity_bytes'],scale)


def compressed(name):
    with gzip.open(EVIDENCE/f'{name}.json.gz','rt',encoding='utf-8') as f:
        return json.load(f)


def tasks(suite):
    cfg = capacity.read(capacity.CONFIG)
    if suite == 'micro':
        models = capacity.models(capacity.read(capacity.R07_CONFIG))
        rows = records('r08-micro')
        for row in rows.values():
            key = row['key']
            model,scale = models[key['model']]
            exp = capacity.open_exp(capacity.micro_fixture(key['fixture']),model,16,key['capacity_bytes'],scale)
            yield row,exp
        return
    if suite == 'design':
        yield from capacity_tasks(cfg,'design',records('r08-design'))
        return
    if suite == 'historical':
        patch = capacity.read(replacement.AMENDMENT)
        trace,old,new = (patch[k] for k in ('trace','exclude_window','replacement_window'))
        # Exactly the corrected published cohort: no extra Coder 96 row.
        cfg['holdout'][trace].remove(old)
        yield from capacity_tasks(cfg,'holdout',records('r08-holdout'))
        cfg['holdout'] = {trace:[new]}
        yield from capacity_tasks(cfg,'holdout',records('r08-replacement'))
        ncfg = capacity.read(native.CFG)
        costs = native.make_costs(ncfg,compressed('r08-native-prefill'),compressed('r08-native-io'))
        rows = records('r08-native')
        for key,data,model in native.tasks(ncfg):
            yield rows[tuple(key.items())], native.get_exp(key,data,model,costs)
        return
    cfg['holdout'] = {B:[481,482]}

    def regression(k,r):
        gap = F(r['main_exact']) - F(r['oracle']['upper_bound'])
        space = F(r['base_exact']) - F(r['oracle']['lower'])
        negative = (k['window']==481 and k['model']=='27b' and k['rate']==16
                    and k['section']=='fixed' and k['level'] in ('176','180','384'))
        return filter_r08(k) or negative or (gap > 1 and 20*gap > space)

    yield from capacity_tasks(cfg,'holdout',records('r08-holdout'),regression)
    diag = capacity.read(EVIDENCE/'r08-counterexample-mechanisms.json')
    keys = [x['key'] for x in diag['r07_unresolved']] + [x['key'] for x in diag['tail_controls'] if 'trace' not in x['key']]
    rows = records('r07-holdout')
    for key,exp in arrivals.cases(arrivals.read(arrivals.CONFIG),'holdout'):
        if key in keys:
            yield rows[tuple(key.items())], exp
    keys = [x['key'] for x in diag['r07_extension']] + [x['key'] for x in diag['tail_controls'] if 'trace' in x['key']]
    rows = records('r07-extension-holdout')
    for key,data,model,scale,_ in extension.cases(capacity.read(extension.CONFIG),'holdout'):
        if key in keys:
            yield rows[tuple(key.items())], extension.experiment(data,key,model,scale)


def run_policy(exp, name, cfg, *, timed=False, audit=False):
    if name != CANDIDATE:
        return arrivals.run(exp,name,timed=timed), {}
    hook = EntryRefiner(exp,cfg,[] if audit else None)
    result = arrivals.run(exp,BASE,timed=timed,extra={'entry_refiner':hook})
    return result, dict(routes=dict(hook.stats), additional_replays=dict(hook.trials), guard_audit=hook.audit)


def profile(row):
    samples = sorted(ns/1e6 for _,ns in row.get('decision_profile_ns',()))
    return dict(total_ms=sum(samples), p50_ms=median(samples),
                p95_ms=samples[ceil(len(samples)*0.95)-1], max_ms=samples[-1]) if samples else None


def compare(exp, result, local, frozen):
    response = arrivals.response(exp,result)
    delta = tuple(a-b for a,b in zip(response,arrivals.response(exp,local),strict=True))
    prior = tuple(a-b for a,b in zip(response,arrivals.response(exp,frozen),strict=True))
    assert all(b in exp.eligible[i] for i,b in enumerate(result['boundaries_tokens']))
    assert result['peak_prefetch_bytes'] <= exp.budget
    return dict(mean=float(result['response_mean_ms']), mean_exact=str(result['response_mean_ms']),
                p95=float(arrivals.p95(response)), decision=profile(result),
                sequence=arrivals.sequence(result), max_local_extra=float(max(0,*delta)),
                max_frozen_extra=float(max(0,*prior)), late=[i for i,x in enumerate(delta) if x>10],
                response_ms=list(map(float,response)), extra_vs_local_ms=list(map(float,delta)),
                extra_vs_frozen_ms=list(map(float,prior)), plan=result['boundaries_tokens'])


def evaluate(exp, row, cfg, rotation, *, ablate):
    base = preflight(exp,row)
    local = arrivals.run(exp,LOCAL)
    assert arrivals.sequence(local) == row['zero'][LOCAL]['sequence'], row['key']
    baseline = {BASE:base, LOCAL:local}
    states = {p:compare(exp,z,local,base) for p,z in baseline.items()}
    policies = {CANDIDATE:cfg}
    if ablate:
        policies.update({k:replace(cfg,**v) for k,v in ABLATIONS.items()})
    for name,p in policies.items():
        z,info = run_policy(exp,CANDIDATE,p,audit=True)
        replay = exp.run('fixed',boundaries=z['boundaries_tokens'])
        assert arrivals.sequence(z) == arrivals.sequence(replay)
        states[name] = dict(compare(exp,z,local,base), **info)
    # Warm each path before measuring; all six execution permutations occur.
    for name in POLICIES:
        run_policy(exp,name,cfg,timed=True)
    orders = tuple(permutations(POLICIES))
    timed = []
    for j in range(len(orders)):
        order = orders[(j+rotation)%len(orders)]
        result = {name:run_policy(exp,name,cfg,timed=True) for name in order}
        paired = {name:dict(compare(exp,z,result[LOCAL][0],result[BASE][0]), **info)
                  for name,(z,info) in result.items()}
        timed.append(dict(order=order, policies=paired))
    repeats = {name:Counter(i for t in timed for i in t['policies'][name]['late']) for name in POLICIES}
    return dict(key=row['key'], archive_exact='main_exact' in row, request_ids=[r['id'] for r in exp.requests], zero=states, timed=timed,
                repeat_tail={p:[i for i,n in sorted(v.items()) if n>=4] for p,v in repeats.items()},
                oracle=row['oracle'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('suite',choices=['micro','design','regression','historical','all'])
    ap.add_argument('--out',required=True,type=Path)
    args = ap.parse_args()
    data = capacity.read(CFG)
    assert data['time_rounds']==6 and data['timed_drop_ms']==10 and data['timed_drop_repeats']==4
    cfg = Refinement(**data['policy'])
    suites = ('micro','design','regression','historical') if args.suite=='all' else (args.suite,)
    opener = gzip.open if args.out.suffix=='.gz' else open
    with opener(args.out,'xt',encoding='utf-8',newline='\n') as dst:
        count = Counter()
        for suite in suites:
            for ix,(r,exp) in enumerate(tasks(suite)):
                value = evaluate(exp,r,cfg,ix,ablate=suite in ('design','regression'))
                dst.write(json.dumps(dict(suite=suite,**value),separators=(',',':'))+'\n'); dst.flush()
                count[suite] += 1
                if ix%36==0:
                    print(suite, ix+1, r['key'], flush=True)
    meta = dict(rows=dict(count),python=platform.python_version(), platform=platform.platform(),
                output=capacity.file_hash(args.out), config=capacity.file_hash(CFG),
                entry_refinement=capacity.file_hash(ROOT/'src/restore_edge_select/entry_refinement.py'),
                simulator=capacity.file_hash(ROOT/'src/restore_edge_select/simulator.py'),
                evaluator=capacity.file_hash(Path(__file__).resolve()))
    with args.out.with_name(args.out.name+'.manifest.json').open('x',encoding='utf-8') as dst:
        json.dump(meta,dst,indent=2); dst.write('\n')
    print(meta, flush=True)


if __name__ == '__main__':
    main()
