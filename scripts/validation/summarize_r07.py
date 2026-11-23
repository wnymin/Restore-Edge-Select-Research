"""Aggregate R07 by frozen cost/arrival strata, resampling source windows."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from random import Random
from statistics import mean, median

from scripts.cpu.replay.study_r07_arrivals import CONFIG, POLICIES, VARIANTS, read

MAIN='entry-replay'
EPS=1e-7


def load(p):
    opener = gzip.open if p.suffix == '.gz' else open
    with opener(p,'rt',encoding='utf-8') as f:
        return [json.loads(s) for s in f]


def ci(values):
    assert len(values) == 12
    rng=Random(707)
    draws=sorted(mean(rng.choices(values,k=len(values))) for _ in range(4000))
    return [draws[100],draws[3899]]


def zero(row):
    upper,lower=row['oracle']['upper'],row['oracle']['lower']
    a,b=(row['zero'][p]['mean'] for p in (MAIN,'local-cost'))
    assert lower <= upper + EPS and upper <= min(a,b) + EPS
    return a,b,lower,upper


def timed(row, metric, comparator='local-cost'):
    return median(t['policies'][MAIN][metric]-t['policies'][comparator][metric] for t in row['timed'])


def summary(records,cfg):
    by_key={(r['key']['rate'],r['key']['variant'],r['key']['window']):r for r in records}
    assert len(by_key)==len(records)==252
    assert all(r['key']['budget']==128 for r in records)
    for row in records:
        assert row['zero'][MAIN]['sequence']==row['zero']['entry']['sequence']
        assert len({tuple(t['order']) for t in row['timed']})==cfg['timing_repeats']==6
        for t in row['timed']:
            assert set(t['order'])==set(t['policies'])==set(POLICIES)
            for p in POLICIES:
                z=t['policies'][p]
                assert z['decision']>0 and len(z['late_indices'])==z['losses_gt10']
                assert abs(z['mean_loss']-(z['mean']-t['policies']['local-cost']['mean']))<EPS
        for v in ('original','even','burst','knee-even','knee-burst'):
            other=by_key[row['key']['rate'],v,row['key']['window']]
            assert other['n']==row['n'] and other['multi']==row['multi']
            factor={'original':1,'even':1,'burst':1,'knee-even':5,'knee-burst':5,'dense':.5,'sparse':2}
            assert abs(other['arrival_span']/factor[v]-row['arrival_span']/factor[row['key']['variant']])<EPS
    table=[]
    compare={'dense':'original','sparse':'original','burst':'even','knee-burst':'knee-even'}
    for rate,_ in cfg['strata']:
        for v in VARIANTS:
            group=[by_key[rate,v,i] for i in cfg['holdout_windows']]
            a,b,l,u=map(mean,zip(*(zero(r) for r in group)))
            loss=[max(0.,a-u),max(0.,a-l)]
            headroom=[max(0.,b-u),max(0.,b-l)]
            capture=[100*(b-a)/h if h>EPS else None for h in headroom[::-1]]
            def material(row):
                a0,b0,l0,u0=zero(row)
                return a0-u0>1 and b0-u0>EPS and (a0-u0)/(b0-u0)>.05
            windows_material=sum(material(r) for r in group)
            if headroom[1]<EPS:
                quality='no-headroom'
            elif loss[0]>1 and headroom[0]>EPS and loss[0]/headroom[0]>.05 and windows_material>=2:
                quality='material-gap'
            elif headroom[0]>EPS and (loss[1]<=1 or loss[1]/headroom[1]<=.05):
                quality='robust-mean'
            else:
                quality='unresolved'
            tmean=[timed(r,'mean') for r in group]; tp95=[timed(r,'p95') for r in group]
            cm,cp=ci(tmean),ci(tp95)
            persistent=[]
            for r in group:
                hit=Counter(i for t in r['timed'] for i in t['policies'][MAIN]['late_indices'])
                if kept:=[j for j,n in hit.items() if n>=4]:
                    persistent.append({'window':r['key']['window'],'requests':sorted(kept)})
            deltas=None
            if v in compare:
                others=[by_key[rate,compare[v],i] for i in cfg['holdout_windows']]
                x,y,p,q=map(mean,zip(*(zero(r) for r in others)))
                # Bounds on changed oracle distance, not on total simulated delay.
                deltas=[loss[0]-max(0.,x-p),loss[1]-max(0.,x-q)]
            table.append(dict(rate=rate,variant=v,cases=len(group),exact=sum(z['oracle']['exact'] for z in group),
                              zero_baseline=b,zero_entry=a,headroom=headroom,capture=capture,gap=loss,
                              material_windows=windows_material,quality=quality,changed_gap=deltas,
                              mean_delta=mean(tmean),mean_ci=cm,p95_delta=mean(tp95),p95_ci=cp,
                              max_single=max(t['policies'][MAIN]['max_loss'] for r in group for t in r['timed']),
                              persistent=persistent,over_budget_p95=cp[0]>5,over_budget_mean=cm[0]>1,
                              timing_p95_status=('degraded' if cp[0]>5 else 'acceptable' if cp[1]<=5 else 'unresolved'),
                              timing_mean_status=('degraded' if cm[0]>1 else 'acceptable' if cm[1]<=1 else 'unresolved'),
                              decision_ms=mean(median(t['policies'][MAIN]['decision'] for t in r['timed']) for r in group),
                              decision_vs_entry_ms=mean(timed(r,'decision','entry') for r in group),
                              max_queue=mean(r['visibility']['max_queue'] for r in group),
                              beyond_four=mean(r['visibility']['beyond_four'] for r in group)))
    return table


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',type=Path); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); result=summary(load(args.data),read(CONFIG))
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for r in result:
        print(r['rate'],r['variant'],r['quality'],'exact',r['exact'],'gap',[round(x,4) for x in r['gap']],
              'cap',[round(x,3) if x is not None else None for x in r['capture']],
              'mean/p95',round(r['mean_delta'],3),round(r['p95_delta'],3),'p95CI',list(map(lambda x:round(x,3),r['p95_ci'])),
              'tail',len(r['persistent']),'queue',round(r['max_queue'],1))


if __name__=='__main__':
    main()
