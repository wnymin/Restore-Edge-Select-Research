"""Fit measured R08 costs and run a separately frozen native-cost replay split."""
import argparse
from collections import defaultdict
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
from statistics import median

from scripts.cpu.replay.study_r08_capacity import (
    MIB, MODEL, ROOT, budgets, cases, dependencies, examine, file_hash, open_exp, paired, read, write_rows,
)
from restore_edge_select.prefill_cost import sampled_cost

CFG=ROOT/'configs/r08-native-calibration.json'


def fit_paths(raw):
    med=defaultdict(dict)
    d=defaultdict(list)
    for r in raw['samples']: d[r['kind'],r['bytes']].append(r['ns'])
    assert len(d)==18 and all(len(v)==raw['rounds'] for v in d.values())
    for (path,size),v in d.items(): med[path][size]=F(median(v),10**6)
    fit={}; stats={}
    mapping={'ram':'ram_to_device','ssd':'ssd_to_device','copy':'device_copy'}
    for path, y in med.items():
        n=len(y); sx=sum(y); sy=sum(y.values())
        slope=(n*sum(x*t for x,t in y.items())-sx*sy)/(n*sum(x*x for x in y)-sx*sx)
        intercept=(sy-slope*sx)/n
        if intercept<0:
            intercept=F(0);slope=sum(x*t for x,t in y.items())/sum(x*x for x in y)
        assert slope>0,path
        fit[mapping[path]]=dict(startup_ms=str(intercept),bytes_per_ms=str(1/slope))
        stats[path]=dict(median_ms={str(x):float(t) for x,t in sorted(y.items())},
                         relative_error={str(x):float((intercept+slope*x-t)/t) for x,t in sorted(y.items())})
    return fit,stats


def make_costs(config, data, io):
    observation={}; grids={}
    for n,d in data['points'].items():
        fit=dict(kind='sampled-prefill',samples=[dict(cached_tokens=r['n_depth'],suffix_tokens=r['n_prompt'],
                     ms=[ns/1_000_000 for ns in r['samples_ns']]) for r in d['grid']])
        pred=sampled_cost(fit)
        observation[n]=[dict(cached=r['n_depth'],suffix=r['n_prompt'],expected_ms=float(pred(r['n_depth'],r['n_prompt'])),
                            measured_ms=median(r['samples_ns'])/1_000_000,
                            relative_error=float(pred(r['n_depth'],r['n_prompt'])/(F(median(r['samples_ns']),10**6))-1))
                        for r in d['observe']]
        assert {(r['cached_tokens'],r['suffix_tokens']) for r in fit['samples']}==set(product(config['depth'],config['suffix']))
        assert {(r['cached'],r['suffix']) for r in observation[n]}==set(product(config['observational_depth'], config['observational_suffix']))
        grids[n]=fit
    fit, stats=fit_paths(io)
    return dict(grids=grids,io=fit,io_residuals=stats,prefill_observe=observation)


def model_cases():
    spec=read(MODEL)['models'][0]
    small=dict(spec,model_id='r08-qwen35-4b-first-16-layers',num_layers=16,
               full_attention_layers_1based=[4,8,12,16],payload={k:v//2 for k,v in spec['payload'].items()})
    return {'4b-native':(spec,F(1)),'16l-native':(small,F(1))}


def tasks(cfg):
    # Reuse input loading but generate the native matrix from its independent split.
    models=model_cases()
    read_cfg=dict(holdout=cfg['heldout'],rates=[16],fixed_mib=[128],reference_tokens=512,
                  matched_padding_mib=1,negative_control_mib=[])
    budget_cfg=dict(read_cfg,fixed_mib=cfg['fixed_mib'])
    for key,data,model,_ in cases(read_cfg,'holdout',models):
        if key['section']!='fixed': continue
        for rate,(section,level,cap) in product(cfg['rates'],budgets(budget_cfg,model)):
            k=dict(key,rate=rate,section=section,level=level,capacity_bytes=cap,cost='measured-both')
            yield k,data,model
        for c in ('measured-compute','measured-io','archived-both'):
            yield dict(key,cost=c),data,model


def get_exp(key, data, model, costs):
    cap,rate=key['capacity_bytes'],key['rate']
    grid=costs['grids'][key['model']] if key['cost'] in ('measured-both','measured-compute') else None
    exp=open_exp(data,model,rate,cap,grid=grid)
    if key['cost'] not in ('measured-both','measured-io'): return exp
    from scripts.cpu.replay.study_r08_capacity import Experiment
    cost=dict(exp.cost)
    for p,coeff in costs['io'].items():
        factor=rate if p=='ssd_to_device' else 1
        cost[p]=dict(startup_ms=str(F(coeff['startup_ms'])*factor),bytes_per_ms=str(F(coeff['bytes_per_ms'])/factor))
    return Experiment(dict(exp.trace,cost=cost), model, F(cap,MIB))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefill',required=True,type=Path);p.add_argument('--io',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--timed-from',type=Path)
    p.add_argument('--calibration-only',action='store_true')
    a=p.parse_args(); cfg=read(CFG)
    costs=make_costs(cfg,read(a.prefill),read(a.io))
    if a.calibration_only:
        with a.out.open('x',encoding='utf-8') as out: json.dump(costs,out,indent=2)
        return
    def output():
        if a.timed_from:
            with a.timed_from.open(encoding='utf-8') as f:
                for ix,((k,d,m),r) in enumerate(zip(tasks(cfg),map(json.loads,f),strict=True)):
                    assert k==r['key']
                    yield dict(key=k,timed=paired(get_exp(k,d,m,costs),cfg['timing_repeats'],ix))
        else:
            for k,d,m in tasks(cfg):
                yield examine(k,get_exp(k,d,m,costs),cfg['node_budget'],diagnosis=True)
    write_rows(output(),a.out)
    manifest=dict(sources={q.relative_to(ROOT).as_posix():file_hash(q) for q in [*dependencies(),CFG,Path(__file__).resolve()]},
                  input_sha256={k:file_hash(v) for k,v in dict(prefill=a.prefill,io=a.io).items()},
                  output_sha256=file_hash(a.out),timed_from_sha256=file_hash(a.timed_from) if a.timed_from else None)
    a.out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
