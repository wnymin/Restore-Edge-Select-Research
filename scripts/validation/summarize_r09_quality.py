"""Produce the R09 paired time/quality curves and per-source acceptance evidence."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import json
from pathlib import Path
from statistics import median

from scripts.cpu.replay.study_r09_budget import capacity, case_id, items, source
from scripts.cpu.replay.study_r09_quality import BASE, LOCAL, quantile


def case(row):
    base,local = row['base'],row['local']
    mb,ml=(F(x['mean_exact']) for x in (base,local))
    lower,upper=(F(row['offline'][k]) for k in ('lower','upper'))
    assert lower<=upper<=mb and upper<=ml
    good = mb==lower or (ml>upper and (mb-lower)*20<=ml-lower)
    assert len(row['timed'])==6
    snapshot={}
    for arm,v in row['snapshot'].items():
        alt=F(v['result']['mean_exact'])
        assert alt>=lower
        ns=[n for _,n in v['extra_ns']]
        capture=None
        if mb>upper:
            capture=sorted([float((mb-alt)/(mb-lower)),float((mb-alt)/(mb-upper))])
        snapshot[arm]=dict(extra_p50_per_hook_ns=quantile(ns,.5),extra_p95_per_hook_ns=quantile(ns,.95),
                           extra_total_ns=sum(ns),changed=len(v['changes']),mean_gain_ms=float(mb-alt),
                           p95_delta_ms=v['result']['p95']-base['p95'],decision_changes=v['changes'],
                           recovered_frozen_gap_fraction=capture,isolated_changed_actions=v['isolated_changed_actions'])
    raw=defaultdict(lambda:defaultdict(list))
    rep=defaultdict(Counter)
    routes=defaultdict(Counter)
    for r in row['timed']:
        arms=r['arms']
        assert len(r['order'])==len(arms) and set(r['order'])==set(arms)==set(row['timed'][0]['arms'])
        a,b=arms[LOCAL],arms[BASE]
        assert len(a['plan'])==len(b['plan'])==len(row['request_ids'])
        ref=Counter(tuple(t) for t in b['own_prefetch_issued'])
        for arm,z in arms.items():
            raw[arm]['mean'].append(z['mean']-b['mean'])
            raw[arm]['p95'].append(z['p95']-b['p95'])
            raw[arm]['decision_p50'].append(z['decision']['p50_ms'])
            raw[arm]['decision_p95'].append(z['decision']['p95_ms'])
            raw[arm]['decision_max'].append(z['decision']['max_ms'])
            raw[arm]['decision_delta_total'].append(z['decision']['total_ms']-b['decision']['total_ms'])
            raw[arm]['queued_compute_gap_delta'].append(z['queued_to_compute_gap_ms']-b['queued_to_compute_gap_ms'])
            raw[arm]['missing_reference_prefetch_events'].append(sum((ref-Counter(tuple(t) for t in z['own_prefetch_issued'])).values()))
            raw[arm]['max_local_extra'].append(z['max_local_extra'])
            if a['mean']>b['mean']:
                raw[arm]['retention'].append((a['mean']-z['mean'])/(a['mean']-b['mean']))
            rep[arm].update(z['late'])
            routes[arm].update(z.get('routes',{}))
    timed={}
    for arm,v in raw.items():
        extra=sorted(i for i,n in rep[arm].items() if n>=4 and rep[BASE][i]<4)
        med={k:median(u) for k,u in v.items()}
        fails=[]
        if good:
            fails += [k for k in ('mean','p95') if med[k]>1]
            if 'retention' in med and med['retention']<.99:
                fails.append('retention')
        if extra:
            fails.append('new_repeat_tail')
        timed[arm]=dict(median=med,max_decision_ms=max(v['decision_max']),routes=routes[arm],
                        new_repeat_indices=extra,failures=fails)
    return dict(key=row['key'],quality=snapshot,timed=timed,offline=row['offline'],known_good=good)


def aggregate(rows):
    quality={}
    timed={}
    for arm in rows[0]['quality']:
        a=[v['quality'][arm] for v in rows]
        ns=[v['extra_total_ns'] for v in a]
        quality[arm]=dict(changed_cases=sum(v['changed']>0 for v in a),
                          zero_mean_improves=sum(v['mean_gain_ms']>1e-8 for v in a),
                          zero_mean_degrades=sum(v['mean_gain_ms']<-1e-8 for v in a),
                          median_total_extra_ns=median(ns),p95_total_extra_ns=quantile(ns,.95),
                          median_p95_hook_ns=median(v['extra_p95_per_hook_ns'] or 0 for v in a),
                          top_gain_ms=max(v['mean_gain_ms'] for v in a))
    for arm in rows[0]['timed']:
        a=[v['timed'][arm] for v in rows]
        counts=Counter(f for v in a for f in v['failures'])
        routes=Counter()
        for v in a:
            routes.update(v['routes'])
        paired_mean=[v['median']['mean'] for v in a]
        source_degradation=defaultdict(list)
        for row in rows:
            source_degradation[source(row['key'])].append(row['timed'][arm]['median']['mean'])
        per_source_worst=[max(x) for x in source_degradation.values()]
        timed[arm]=dict(case_failures=counts,routes=routes,mean_faster_over_1ms=sum(x<-1 for x in paired_mean),
                        worst_case_mean=max(paired_mean),worst_case_p95=max(v['median']['p95'] for v in a),
                        median_case_mean=median(paired_mean),median_case_p95=median(v['median']['p95'] for v in a),
                        median_case_decision_p95_ms=median(v['median']['decision_p95'] for v in a),
                        source_worst_mean_observed_range_ms=[min(per_source_worst),max(per_source_worst)],
                        source_worst_mean_observed_median_ms=median(per_source_worst),
                        max_actual_decision_ms=max(v['max_decision_ms'] for v in a))
    return dict(configurations=len(rows),source_windows=len({source(r['key']) for r in rows}),
                exact_oracle=sum(r['offline']['exact'] for r in rows),known_good=sum(r['known_good'] for r in rows),
                snapshot=quality,timed=timed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('inputs','budget_lock','raw','out'):
        p.add_argument(k,type=Path)
    p.add_argument('--split',choices=('calibration','holdout'),default='holdout')
    a=p.parse_args()
    m=capacity.read(a.inputs)
    assert capacity.read(a.budget_lock)['inputs_sha256']==capacity.file_hash(a.inputs)
    data=[case(r) for r in items(a.raw,m,a.split)]
    groups=defaultdict(list)
    for r in data:
        groups[f"{r['key']['model']}/{r['key']['stratum']}"] .append(r)
    output=dict(split=a.split,inputs_sha256=capacity.file_hash(a.inputs),lock_sha256=capacity.file_hash(a.budget_lock),
                raw_sha256=capacity.file_hash(a.raw), total=aggregate(data),
                groups={k:aggregate(v) for k,v in sorted(groups.items())},cases=data)
    with a.out.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(output,f,indent=2,allow_nan=False)
        f.write('\n')


if __name__=='__main__':
    main()
