"""R09 supplemental fixed-action critical-point audit; never reselects a budget."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path

from scripts.cpu.replay.study_r09_budget import assess_rule, by_window, capacity, case_id, items, workloads
from scripts.cpu.replay.study_r09_slack import effects, fixed, prefix_cap, starts


def refine_points(grid, state):
    points = {F(x['delay_ms']) for x in grid}
    for metric in ('events_safe','perf_safe'):
        lo,hi = prefix_cap(grid,metric)
        if hi is None:
            points.update(map(F,(64,128,256,512)))
        else:
            points.update(lo+(hi-lo)*i/4 for i in (1,2,3))
    remaining = state['running_until_ms']
    if remaining is not None:
        residual=F(remaining)-F(state['publish_ms'])
        if residual>0:
            points.update(d for d in (residual-F('.05'),residual,residual+F('.05')) if 0<d<=512)
    return sorted(points)


def timeline(exp,row,result,ix):
    state=result['decision_timing'][ix]
    deps=set(exp.candidates[ix][row['plan'][ix]])
    rid=result['request_ids'][ix]
    return dict(state=state,shared_charged_candidate_bytes=exp.size(deps-exp.initial)-state['added_bytes'],
                gpu_compute_start_ms=starts(exp,result)[ix],request_completion_ms=result['completion_ms'][ix],
                reserved_at_ms=result['prefetch_reserved_at_ms'][ix],
                own_transfers=[t for t in result['transfers'] if t['request']==rid],
                dependency_transfers=[t for t in result['transfers'] if t['object'] in deps and t['mode']!='handoff'])


def analyze(manifest,cal,hold):
    picked={}
    primary={}
    for row in items(hold,manifest,'holdout'):
        k=row['key']
        if k['model']!='4b-native':
            continue
        primary[case_id(k)]=row
        ds=[d for d in row['decisions'] if d['state']['eligible_count']>1]
        for label,pred in (
            ('earliest-zero-floor-while-gpu-busy',lambda d:d['stratum'].startswith('busy/') and prefix_cap(d['grid'],'perf_safe')[0]==0),
            ('earliest-right-censored-while-gpu-busy',lambda d:d['stratum'].startswith('busy/') and prefix_cap(d['grid'],'perf_safe')[1] is None),
        ):
            d=next((x for x in ds if pred(x)),None)
            if d:
                picked.setdefault(case_id(k),(row,[]))[1].append((label,d))
    for row in items(cal,manifest,'calibration'):
        k=row['key']
        pair=(k['window'],k['model'],k['rate'],k['capacity_bytes']//2**20)
        targets={(481,'4b',12,176):7,(482,'27b',16,384):16,(481,'27b',16,512):17}
        if k['stratum']=='proxy/witness' and pair in targets:
            d=row['decisions'][targets[pair]]
            picked[case_id(k)]=(row,[('R08-regression-witness',d)])
    samples=[]
    for key,exp in workloads(dict(cases=[v[0]['key'] for v in picked.values()],strata=manifest['strata'])):
        row,decisions=picked[case_id(key)]
        duration=[F(n,1_000_000) for n in row['sample_ns']]
        original=fixed(exp,row['plan'],duration,{})
        for label,d in decisions:
            ix=d['state']['index']
            assert str(original['decision_timing'][ix]['publish_ms']) == d['state']['publish_ms']
            published=[]
            for delay in refine_points(d['grid'],d['state']):
                trial = fixed(exp,row['plan'],duration,{ix:delay})
                e = effects(exp,original,trial)
                published.append(dict(delay_ms=str(delay),**e))
            e0,e1=prefix_cap(published,'events_safe'),prefix_cap(published,'perf_safe')
            samples.append(dict(case=key,label=label,index=ix,decision_stratum=d['stratum'],
                                reference=timeline(exp,row,original,ix),
                                conservative_bracket=e0,performance_bracket=e1,scanned=published))
    smallest = [assess_rule(exp,primary[case_id(key)],F('.05')) for key,exp in workloads(
        dict(cases=[r['key'] for r in primary.values()],strata=manifest['strata']))]
    return dict(smallest_positive_main_grid_dynamic_busy=dict(cap_ms='.05',cases=smallest,windows=by_window(smallest)),
                method='Outcome-stratified diagnostic subset, not a budget lock, prevalence estimate, interpolation, or proof between tested points.',
                calibration_sha256=capacity.file_hash(cal),holdout_sha256=capacity.file_hash(hold),
                independent_source_claim=False,selected_configs=len(picked),samples=samples)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','calibration','holdout','out'):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    data=analyze(capacity.read(a.inputs),a.calibration,a.holdout)
    with a.out.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(data,f,indent=2,default=str)
        f.write('\n')


if __name__=='__main__':
    main()
