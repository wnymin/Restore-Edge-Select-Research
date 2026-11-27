"""R09 source-unit budget calibration and descriptive public holdout verification."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from fractions import Fraction as F
import gzip
from hashlib import sha256
import json
from math import ceil
from pathlib import Path

from scripts.cpu.replay.study_r09_slack import effects, prefix_cap
from scripts.validation.archive_source import source_matches
from scripts.validation.run_r08_sealed_holdout import capacity, excluded, workloads

ROOT = capacity.ROOT
EVIDENCE = ROOT/'docs/evidence'
SOURCES = ('src/restore_edge_select/simulator.py','src/restore_edge_select/entry_refinement.py',
           'scripts/cpu/replay/study_r09_slack.py','scripts/cpu/replay/study_r09_quality.py',
           'scripts/cpu/replay/study_r09_budget.py','configs/r08-entry-refinement.json')


def source_pin(path):
    return sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def caseset(manifest, split):
    return {case_id(c):c for c in manifest[split]}


def case_id(c):
    return (c['trace'],c['window'],c['model'],c['rate'],c['capacity_bytes'],c['stratum'])


def source(c):
    return c['trace'],c['window']


def items(raw,manifest,split):
    expected = caseset(manifest,split)
    assert len(expected)==len(manifest[split])
    with gzip.open(raw,'rt',encoding='utf-8') as stream:
        for row in map(json.loads,stream):
            assert expected.pop(case_id(row['key']))==row['key']
            yield row
    assert not expected


def check_inputs(manifest):
    older = excluded() | {source(c) for c in capacity.read(EVIDENCE/'r08-takeover-seal.json')['cases']}
    cal, hold = ({source(a) for a in manifest[s]} for s in ('calibration','holdout'))
    assert not (older | cal) & hold
    assert len(hold)==18
    for trace in manifest['trace_sha256']:
        w = sorted(w for t,w in hold if t==trace)
        assert all(j-i >= manifest['min_window_index_spacing'] for i,j in zip(w,w[1:]))
        assert capacity.file_hash(capacity.TRACE_DIR/trace)==manifest['trace_sha256'][trace]
    for a in [*manifest['calibration'],*manifest['holdout']]:
        assert a['raw_rows']==[352*a['window'],352*(a['window']+1)]
        assert a['source_sha256']==manifest['trace_sha256'][a['trace']]


def budget_rule(exp, plan, ns, cap):
    """Charge every eligible busy decision, evaluated at its actual replay time."""
    budget = F(cap)
    def rule(state):
        enabled = state['eligible_count']>1 and state['head']<state['index'] and state['running_until_ms'] is not None
        return F(ns[state['index']],1_000_000) + (budget if enabled else 0)
    return exp.run('fixed',boundaries=plan,decision_timing=rule)


def assess_rule(exp,row,cap,origin=None):
    base = origin if origin is not None else budget_rule(exp,row['plan'],row['sample_ns'],0)
    out = budget_rule(exp,row['plan'],row['sample_ns'],cap) if cap else base
    enabled = [s['index'] for s in out['decision_timing'] if s['publish_ms']-s['begin_ms']>F(row['sample_ns'][s['index']],1_000_000)]
    return dict(case=row['key'],cap_ms=str(cap),enabled=enabled,**effects(exp,base,out))


def by_window(rows):
    out = {}
    for row in rows:
        a=out.setdefault(source(row['case']),dict(failed=False,events_failed=False,enabled=0))
        a['failed'] |= not row['perf_safe']
        a['events_failed'] |= not row['events_safe']
        a['enabled'] += len(row['enabled'])
    return [dict(trace=t,window=w,**a) for (t,w),a in sorted(out.items())]


def primary(manifest, split):
    return [a for a in manifest[split] if a['model']=='4b-native']


def lock(manifest, raw, args):
    check_inputs(manifest)
    rows={case_id(r['key']):r for r in items(raw,manifest,'calibration')}
    grid=[F(v) for v in manifest['grid_ms']]
    evidence = defaultdict(list)
    for key,exp in workloads(dict(cases=primary(manifest,'calibration'),strata=manifest['strata'])):
        row=rows[case_id(key)]
        origin=budget_rule(exp,row['plan'],row['sample_ns'],0)
        for cap in grid:
            evidence[cap].append(assess_rule(exp,row,cap,origin=origin))
    table = {str(cap):by_window(rs) for cap,rs in evidence.items()}
    selected = F(0)
    for b in grid[1:]:
        v = table[str(b)]
        if any(x['failed'] for x in v) or not any(x['enabled'] for x in v):
            break
        selected = b
    return dict(schema=1,locked_at=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds'),
                inputs_sha256=capacity.file_hash(args.inputs),calibration_sha256=capacity.file_hash(raw),
                pins={k:source_pin(ROOT/k) for k in SOURCES}, primary_model='4b-native',
                rule='eligible multi-choice prefetch decision with an active computation at actual decision start; otherwise zero',
                selection='Largest cap in consecutively passing tested grid prefix with zero worse-performance calibration source windows and positive usage. If none, zero.',
                release_budget_ms=str(selected),calibration_source_checks=table,
                experimental_caps_ms=list(dict.fromkeys(['0.25','2','8']+([str(selected)] if selected and selected not in (F('.25'),2,8) else []))),
                step_guard_ns=50_000,oracle_nodes=1024,confidence=.95,max_failed_window_rate=.05,
                inferential_status='descriptive shared public traces; no independent-source probability certification')


def quantile(v,p):
    return str(sorted(v)[ceil(len(v)*p)-1]) if v else None


def describe(slack_rows):
    states = defaultdict(list)
    shared = defaultdict(list)
    for row in slack_rows:
        key=row['key']; grouping=key['model'],key['stratum']
        shared[grouping].append(row)
        for d in row['decisions']:
            if d['state']['eligible_count']>1:
                states[(*grouping,d['stratum'])].append((source(key),d))
    stats = []
    for (model,condition,state),group in sorted(states.items()):
        out=dict(model=model,condition=condition,state=state,decisions=len(group),
                 sources=len({t for (t,_),_ in group}),windows=len({s for s,_ in group}))
        for label,metric in (('conservative','events_safe'),('performance','perf_safe')):
            answers=[prefix_cap(d['grid'],metric) for _,d in group]
            v=[a for a,_ in answers]
            rem=[(a,F(d['state']['running_until_ms'])-F(d['state']['publish_ms'])) for (_,d),a in zip(group,v,strict=True)
                 if d['state']['running_until_ms'] is not None and F(d['state']['running_until_ms'])>F(d['state']['publish_ms'])]
            out[label]=dict(zero_floor=sum(x==0 for x in v),right_censored=sum(b is None for _,b in answers),
                            nonmonotonic=sum(any(rr[metric] for rr in d['grid'][next((i for i,r in enumerate(d['grid']) if not r[metric]),len(d['grid'])):]) for _,d in group),
                            p05=quantile(v,.05),p10=quantile(v,.10),p50=quantile(v,.5),p95=quantile(v,.95),
                            projected_gpu_overlap_n=len(rem),
                            projected_gpu_overlap_share_floor_p50=quantile([min(1,a/b) for a,b in rem],.5))
        stats.append(out)
    group_grid=[]
    for (model,condition),group in sorted(shared.items()):
        grid=[]
        for i,d in enumerate(group[0]['joint']):
            grid.append(dict(delay_ms=d['delay_ms'],total=len(group),
                all_bad=sum(not r['joint'][i]['perf_safe'] for r in group),
                busy_bad=sum(not r['joint_busy'][i]['perf_safe'] for r in group),
                all_event_bad=sum(not r['joint'][i]['events_safe'] for r in group),
                busy_event_bad=sum(not r['joint_busy'][i]['events_safe'] for r in group)))
        group_grid.append(dict(model=model,condition=condition,grid=grid))
    return dict(single=stats,joint=group_grid)


def verify(manifest,raw,args):
    check_inputs(manifest)
    seal=capacity.read(args.lock)
    assert seal['inputs_sha256']==capacity.file_hash(args.inputs)
    assert all(source_matches(ROOT/k,pin,historical=True) for k,pin in seal['pins'].items())
    data=list(items(raw,manifest,'holdout'))
    rows={case_id(r['key']):r for r in data}
    chosen=F(seal['release_budget_ms'])
    caps=sorted({chosen,*map(F,seal['experimental_caps_ms'])})
    checks=defaultdict(list)
    for key,exp in workloads(dict(cases=primary(manifest,'holdout'),strata=manifest['strata'])):
        row=rows[case_id(key)]
        origin=budget_rule(exp,row['plan'],row['sample_ns'],0)
        for cap in caps:
            checks[str(cap)].append(assess_rule(exp,row,cap,origin=origin))
    return dict(inputs_sha256=seal['inputs_sha256'],budget_lock_sha256=capacity.file_hash(args.lock),
                holdout_sha256=capacity.file_hash(raw),status=seal['inferential_status'],deployment_extra_ms='0',
                nominal_zero_failure_upper_if_independent=1-.05**(1/18),selected_calibration_cap_ms=str(chosen),
                sample=dict(cases=len(data),source_windows=len({source(a['key']) for a in data})),
                descriptive=describe(data),primary={b:dict(windows=by_window(v),cases=v) for b,v in checks.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('lock','verify'))
    p.add_argument('inputs',type=Path)
    p.add_argument('raw',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('--lock',type=Path)
    a=p.parse_args()
    f=lock if a.stage=='lock' else verify
    val=f(capacity.read(a.inputs),a.raw,a)
    with a.out.open('x',encoding='utf-8',newline='\n') as dst:
        json.dump(val,dst,indent=2)
        dst.write('\n')


if __name__=='__main__':
    main()
