"""Run the frozen R08 candidate on a collaborator-supplied, already-revealed seal.

This tool does not choose windows, certify independence, or issue a passing
verdict. The collaborator keeps the manifest private until the agreed reveal.
"""
import argparse
from collections import defaultdict
from fractions import Fraction as F
import gzip
import json
from pathlib import Path

from scripts.cpu.replay.study_r08_refinement import (
    BASE, CANDIDATE, CFG, LOCAL, ROOT, Refinement, arrivals, capacity, compressed, evaluate, native, run_policy,
)
from scripts.validation.archive_source import source_matches
from scripts.validation.audit_r08_source_windows import keys, streams
from restore_edge_select.offline_bounds import search

FILES = dict(config=CFG, entry_refinement=ROOT/'src/restore_edge_select/entry_refinement.py',
             simulator=ROOT/'src/restore_edge_select/simulator.py',
             evaluator=ROOT/'scripts/cpu/replay/study_r08_refinement.py')
STRESSES = {'low-no-room','near-optimal','capacity-edge','high-pressure'}
EVIDENCE = ROOT/'docs/evidence'


def excluded():
    out = {k for name,raw in streams() for k in keys(name,raw)}
    c=capacity.read(ROOT/'configs/r07-arrival-matrix.json')
    out.update(('qwen_traceB_blksz_16.jsonl',w) for name in ('design_windows','holdout_windows') for w in c[name])
    for path,sections in ((capacity.R07_CONFIG,('design','holdout')),
                          (capacity.CONFIG,('design','holdout')),(native.CFG,('heldout',))):
        c=capacity.read(path)
        out.update((t,w) for section in sections for t,ws in c[section].items() for w in ws)
    x=capacity.read(ROOT/'configs/r08-capacity-holdout-amendment.json')
    out.update((x['trace'],x[k]) for k in ('exclude_window','replacement_window'))
    return out


def validate(m, old_windows, expected_pins):
    required={'schema','auditor','sealed_at','frozen_sha256','driver_sha256','thresholds','interval_method','oracle_nodes','strata','cases'}
    if set(m)!=required or m['schema']!=1:
        raise ValueError('unexpected manifest schema')
    for k in ('auditor','sealed_at','interval_method'):
        if not isinstance(m[k],str) or not m[k].strip():
            raise ValueError(f'missing auditor metadata: {k}')
    if m['frozen_sha256']!=expected_pins or not all(source_matches(FILES[k],v) for k,v in expected_pins.items()):
        raise ValueError('frozen candidate or baseline source pin mismatch')
    if not source_matches(Path(__file__).resolve(),m['driver_sha256']):
        raise ValueError('timing driver pin mismatch')
    if m['thresholds'] != dict(mean_ms=1,p95_ms=1,keep_net=0.99,gap_fraction=0.05,
                              recover_fraction=0.5,extra_request_ms=10,repeat_excess=4,repeats=6):
        raise ValueError('changed dual-gate thresholds')
    if type(m['oracle_nodes']) is not int or m['oracle_nodes']<1:
        raise ValueError('invalid offline search budget')
    counts=defaultdict(set); identities=set(); planned={}
    for label,s in m['strata'].items():
        if set(s)!={'pressure','cost_domain','source_windows'} or s['pressure'] not in STRESSES or s['cost_domain'] not in ('proxy','measured'):
            raise ValueError(f'invalid stratum {label}')
        if type(s['source_windows']) is not int or s['source_windows']<2:
            raise ValueError(f'insufficient prespecified source windows: {label}')
        planned.setdefault(s['cost_domain'],set()).add(s['pressure'])
    if set(planned)!={'proxy','measured'} or set.union(*planned.values())!=STRESSES:
        raise ValueError('plan must cover four pressure classes with separate proxy and measured strata')
    model_map=capacity.models(capacity.read(capacity.R07_CONFIG))
    sources={x['name']:x for x in capacity.read(capacity.MANIFEST)['files']}
    already_checked=set()
    for a in m['cases']:
        if set(a)!={'trace','window','raw_rows','source_sha256','model','rate','capacity_bytes','stratum'}:
            raise ValueError('unexpected case keys')
        t,w=a['trace'],a['window']; d=m['strata'][a['stratum']]['cost_domain']
        if t not in sources or type(w) is not int or w<0 or a['raw_rows']!=[352*w,352*(w+1)]:
            raise ValueError('expected pinned 352-row half-open source range')
        if (t,w) in old_windows:
            raise ValueError('overlapping historical source window')
        if a['source_sha256']!=sources[t]['sha256']:
            raise ValueError('source pin mismatch')
        if t not in already_checked:
            source=capacity.TRACE_DIR/t
            if source.stat().st_size!=sources[t]['bytes'] or capacity.file_hash(source)!=sources[t]['sha256']:
                raise ValueError('local trace differs from the committed source')
            already_checked.add(t)
        if a['model'] not in (model_map if d=='proxy' else native.model_cases()):
            raise ValueError('profile and cost domain mismatch')
        if type(a['capacity_bytes']) is not int or not 0<=a['capacity_bytes']<=512*capacity.MIB or type(a['rate']) is not int or a['rate']<1:
            raise ValueError('invalid capacity or pressure rate')
        identity=(t,w,d,a['model'],a['rate'],a['capacity_bytes'])
        if identity in identities:
            raise ValueError('duplicated case across one or more strata')
        identities.add(identity); counts[a['stratum']].add((t,w))
    if {k:len(v) for k,v in counts.items()}!={k:s['source_windows'] for k,s in m['strata'].items()}:
        raise ValueError('actual source-window strata do not match the seal')


def trial(config, exp, identity, nodes, rotation):
    local,base=(arrivals.run(exp,p) for p in (LOCAL,BASE))
    candidate,_=run_policy(exp,CANDIDATE,config)
    lo,hi,plan,visits=search(exp,[local,base,candidate],nodes,resource_floor=True,dominance=True)
    n=len(exp.requests)
    assert exp.run('fixed',boundaries=plan)['response_sum_ms']==hi
    row=dict(key=identity,main_exact=str(base['response_mean_ms']),zero={p:{'sequence':arrivals.sequence(z)}
        for p,z in ((LOCAL,local),(BASE,base))},oracle=dict(lower=str(lo/n),upper_bound=str(hi/n),upper=float(hi/n),
        exact=lo==hi,nodes=visits,plan=plan))
    return evaluate(exp,row,config,rotation,ablate=False)


def workloads(m):
    ref=capacity.read(capacity.MODEL)['models'][0]; fixture=capacity.read(capacity.FIXTURE)
    proxy=capacity.models(capacity.read(capacity.R07_CONFIG)); measured=native.model_cases()
    modelcost=native.make_costs(capacity.read(native.CFG),compressed('r08-native-prefill'),compressed('r08-native-io'))
    for t in sorted({a['trace'] for a in m['cases']}):
        group=defaultdict(list)
        for a in m['cases']:
            if a['trace']==t:
                group[a['window']].append(a)
        for w,_,_,spec in capacity.window_data(capacity.TRACE_DIR,t,group,fixture,ref,[('split',F(1))]):
            for a in group[w]:
                domain=m['strata'][a['stratum']]['cost_domain']
                if domain=='measured':
                    exp=native.get_exp(dict(a,cost='measured-both'),spec,measured[a['model']][0],modelcost)
                else:
                    model,scale=proxy[a['model']]
                    exp=capacity.open_exp(spec,model,a['rate'],a['capacity_bytes'],scale)
                yield a,exp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path); p.add_argument('--sealed-sha256',required=True)
    p.add_argument('--out',required=True,type=Path); args=p.parse_args()
    actual=capacity.file_hash(args.manifest)
    if actual!=args.sealed_sha256.lower():
        p.error('revealed manifest bytes do not match the externally recorded seal')
    m=capacity.read(args.manifest)
    pins=capacity.read(EVIDENCE/'r08-entry-refinement-selftest.json')['provenance']
    validate(m,excluded(),{k:pins[k] for k in FILES})
    cfg=Refinement(**capacity.read(CFG)['policy'])
    op=gzip.open if args.out.suffix=='.gz' else open
    with op(args.out,'xt',encoding='utf-8',newline='\n') as stream:
        for i,(a,exp) in enumerate(workloads(m)):
            value=trial(cfg,exp,a,m['oracle_nodes'],i)
            stream.write(json.dumps(value,separators=(',',':'))+'\n')
            stream.flush()
    assert i+1==len(m['cases'])
    meta=dict(sealed_sha256=actual, output_sha256=capacity.file_hash(args.out), n=i+1,
              frozen_sha256=m['frozen_sha256'],driver_sha256=m['driver_sha256'])
    with args.out.with_name(args.out.name+'.manifest.json').open('x',encoding='utf-8') as f:
        json.dump(meta,f,indent=2); f.write('\n')


if __name__=='__main__':
    main()
