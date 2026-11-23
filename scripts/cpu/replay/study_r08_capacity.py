"""R08: fixed and matched-capacity profiles through the unchanged R07 policy."""
import argparse
from collections import Counter
from copy import deepcopy
from fractions import Fraction as F
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import platform

from scripts.cpu.replay.study_r07_extension import (
    CONFIG as R07_CONFIG, FIXTURE, GRID, IO, MODEL, MANIFEST, OVERLAP, ROOT, SYNTH,
    TRACE_DIR, Experiment, change_payload, file_hash, models, paired, pack, read,
    response, run, sequence, window_data, write_rows,
)
from scripts.cpu.dynamic.dynamic_cpu_inputs import make_experiment
from scripts.cpu.replay.study_r07_diagnostics import diagnose
from restore_edge_select.offline_bounds import search

CONFIG = ROOT / 'configs/r08-capacity-matrix.json'
MIB = 2**20
POLICIES = ('local-cost', 'entry', 'entry-replay')


def references(model, tokens):
    p = model['payload']
    one = tokens * p['attention_bytes_per_token'] + p['checkpoint_bytes']
    return one, one * 2


def budgets(cfg, model):
    for m in cfg['fixed_mib']:
        yield 'fixed', str(m), m*MIB
    first, pair = references(model, cfg['reference_tokens'])
    for label, need in [('one', first), ('two-private', pair)]:
        for side, direction in [('below', -1), ('above', 1)]:
            yield 'matched', f'{label}-{side}', need+direction*cfg['matched_padding_mib']*MIB


def open_exp(spec, model, rate, cap, scale=F(1), grid=None):
    mem = change_payload(spec, model)
    m = mem['memory']
    assert cap <= 512*MIB
    mem = dict(mem, memory=dict(m, device_mib=m['fixed_mib']+m['execution_mib']+512))
    exp = make_experiment(read(FIXTURE), model, mem, 'ssd', rate, F(cap,MIB))
    if grid is not None or scale != 1:
        if grid is None:
            grid = exp.cost['recompute']
            grid = dict(grid, samples=[dict(row, ms=[str(F(str(x))*scale) for x in row['ms']])
                                       for row in grid['samples']])
        exp = Experiment(dict(exp.trace,cost=dict(exp.cost,recompute=grid)),model,F(cap,MIB))
    return exp


def cases(cfg, split, model_map=None):
    manifest = {e['name']:e for e in read(MANIFEST)['files']}
    fixture = read(FIXTURE)
    anchor = read(MODEL)['models'][0]
    model_map = model_map or {m: v for m,v in models(read(R07_CONFIG)).items() if m in cfg['main_models']}
    for trace, windows in cfg[split].items():
        path=TRACE_DIR/trace
        if path.stat().st_size!=manifest[trace]['bytes'] or file_hash(path)!=manifest[trace]['sha256']:
            raise ValueError(f'Unpinned source {trace}')
        for window,_,_,data in window_data(TRACE_DIR,trace,windows,fixture,anchor,[('split',F(1))]):
            for label, (model, scale) in model_map.items():
                strata=[(r,*x) for r,x in product(cfg['rates'],budgets(cfg,model))]
                strata += [(1,'control',str(b),b*MIB) for b in cfg['negative_control_mib']]
                for rate,section,level,cap in strata:
                    key=dict(trace=trace,window=window,model=label,rate=rate,section=section,level=level,capacity_bytes=cap)
                    yield key, data, model, scale


def examine(key, exp, node_budget, *, exhaustive=False, diagnosis=False):
    z={p:run(exp,p) for p in POLICIES}
    base,main=z['local-cost'],z['entry-replay']
    assert sequence(main)==sequence(z['entry']),key
    assert exp.run('fixed',boundaries=main['boundaries_tokens'])['completion_ms']==main['completion_ms']
    if exhaustive:
        o=exp.offline()
        lo=hi=o['response_sum_ms']; plan=o['boundaries_tokens']; visited=None
    else:
        lo,hi,plan,visited=search(exp,list(z.values()),node_budget,resource_floor=True,dominance=True)
    assert lo<=hi<=min(base['response_sum_ms'],main['response_sum_ms'])
    assert exp.run('fixed',boundaries=plan)['response_sum_ms']==hi
    n=len(exp.requests)
    out=dict(key=key,n=n, base_exact=str(base['response_mean_ms']),main_exact=str(main['response_mean_ms']),
             oracle=dict(lower=str(lo/n),upper_bound=str(hi/n),upper=float(hi/n),exact=(lo==hi),nodes=visited,plan=plan),
             zero={p:pack(exp,r,response(exp,base)) for p,r in z.items()},plan=main['boundaries_tokens'],
             feasible_by_size=dict(total=sum(len(e)-1 for e in exp.eligible),
                 cold=sum(bool(ds) and exp.size(ds)<=exp.budget for x in exp.candidates for ds in x.values())),
             activity={p:dict(peak=r['peak_prefetch_bytes'],
                              reserved=sum(t is not None for t in r['prefetch_reserved_at_ms']),
                              bytes_by_path=r['transferred_bytes_by_path']) for p,r in z.items()},
             input=dict(order=sha256(repr((exp.arrivals,exp.lengths,exp.eligible,
                             [x['id'] for x in exp.requests])).encode()).hexdigest(),
                        execution=exp.execution,window=exp.window))
    if diagnosis and hi<main['response_sum_ms']:
        w=diagnose(exp,out)
        out['diagnostic']=w[0] if isinstance(w,tuple) else w
    return out


def micro_fixture(label):
    f=read(FIXTURE)
    if label in ('stagger','late'):
        f['requests'][2]['arrival_ms'] = 2 if label=='stagger' else 10000
    if label=='shared-kv':
        c=f['requests'][2]
        old=c['input_token_runs'][0]['token_id']
        new=f['requests'][1]['input_token_runs'][0]['token_id']
        for r in [c['input_token_runs'],*(o['prefix_token_runs'] for o in f['objects'] if o['id'].startswith('C.'))]:
            for t in r:
                if t['token_id']==old: t['token_id']=new
        for m in c['candidates']:
            m['load_order']=['B.kv1' if oid=='C.kv1' else oid for oid in m['load_order']]
    return f


def micro_cases(cfg):
    maps=models(read(R07_CONFIG))
    for name, variant in product(cfg['main_models'],('same-time','stagger','shared-kv','late')):
        model,scale=maps[name]
        trace=micro_fixture(variant)
        reference=open_exp(trace,model,16,0,scale)
        a,b=[set(cs[576]) for cs in reference.candidates[1:]]
        less=references(model,576)[0]
        union=less*2-(576*model['payload']['attention_bytes_per_token'] if variant=='shared-kv' else 0)
        assert a and b and reference.size(a)==reference.size(b)==less and reference.size(a|b)==union
        for (target,need),offset in product([('one',less),('union',union)],cfg['micro_offsets_bytes']):
            cap=need+offset
            key=dict(model=name,fixture=variant,target=target,offset_bytes=offset,capacity_bytes=cap)
            exp=open_exp(trace,model,16,cap,scale)
            z=examine(key,exp,cfg['node_budget'],exhaustive=True)
            fixed=exp.run('fixed',boundaries=(0,576,576))
            actual=fixed['prefetch_reserved_at_ms']
            first=actual[1]==exp.arrivals[1]
            second=actual[2]==exp.arrivals[2]
            assert first==(cap>=less),(key, actual)
            assert second==(variant!='late' and cap>=union),(key, actual)
            z['fixed_witness']=dict(plan=[0,576,576],reservations=[str(t) if t is not None else None for t in actual],
                first_at_arrival=first,second_at_arrival=second,peak=fixed['peak_prefetch_bytes'],
                completion=[str(t) for t in fixed['completion_ms']],threshold_bytes=need,
                one_bytes=less,union_bytes=union,shared_ids=sorted(a&b))
            yield z


def dependencies():
    return [Path(__file__).resolve(),CONFIG,FIXTURE,GRID,IO,MODEL,MANIFEST,OVERLAP,SYNTH,R07_CONFIG,
            *(ROOT/f'scripts/cpu/replay/{s}.py' for s in ('study_r07_extension','study_r07_arrivals',
                'study_r07_diagnostics','study_bailian','study_boundary_factors')),
            ROOT/'scripts/cpu/dynamic/dynamic_cpu_inputs.py',*(ROOT/'src/restore_edge_select').glob('*.py')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('split',choices=['micro','design','holdout','timed'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--untimed',type=Path)
    a=p.parse_args()
    cfg=read(CONFIG)
    if a.split=='micro':
        output=micro_cases(cfg)
    elif a.split=='timed':
        if a.untimed is None: p.error('--untimed required')
        def timed():
            with a.untimed.open(encoding='utf-8') as f:
                for ix,(task,old) in enumerate(zip(cases(cfg,'holdout'),map(json.loads,f),strict=True)):
                    key,data,model,scale=task
                    assert key==old['key']
                    yield dict(key=key,timed=paired(open_exp(data,model,key['rate'],key['capacity_bytes'],scale),
                                                  cfg['timing_repeats'],ix))
        output=timed()
    else:
        def zero():
            for key,data,model,scale in cases(cfg,a.split):
                exp=open_exp(data,model,key['rate'],key['capacity_bytes'],scale)
                yield examine(key,exp,cfg['node_budget'],diagnosis=a.split=='holdout')
        output=zero()
    write_rows(output,a.out)
    meta=dict(split=a.split,platform=platform.platform(),python=platform.python_version(),
              input_sha256=file_hash(a.untimed) if a.untimed else None,output_sha256=file_hash(a.out),
              files={f.relative_to(ROOT).as_posix():file_hash(f) for f in dependencies()})
    a.out.with_suffix('.manifest.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
