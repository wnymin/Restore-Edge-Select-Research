"""Linear-size analytic prefix certificate for demand-only vs recompute."""

import argparse
from fractions import Fraction as F
import json
from pathlib import Path
from random import Random

from scripts.cpu.lookahead.robustness.study_lookahead_joint_domain import Capture, Linear, ROOT, Region, Split, analyze, oracle, read, sign
from scripts.cpu.lookahead.staging.study_lookahead_three_request import profile
from scripts.cpu.lookahead.staging.study_lookahead_two_request import jsonable


def certificate(exp,record,box):
    target,fallback,state,base,estimate,speeds=record
    assert (state[1],target,state[2],fallback)==(0,1,3,0)
    assert not exp.per_tier and not exp.dynamic and not exp.host_cache_bytes
    assert exp.release_at=='start' and exp.window>=2
    assert state[5]>state[0]
    assert state[4]==exp.initial and not state[8] and state[9] is None and not state[10]
    assert max(exp.arrivals)<=state[0]
    choices=exp.eligible[1]
    assert len(choices)==2 and choices[0]==0
    deep=choices[1]
    p=profile(exp,1,deep,estimate,speeds)
    zero=profile(exp,1,0,estimate,speeds)
    later=profile(exp,2,base[2],estimate,speeds)
    inflight=state[6]
    if inflight is None:
        assert not state[3]
    else:
        assert state[3]=={2:set(later.missing)}
        assert inflight[0]>state[0] and inflight[1] in later.missing
    assert exp.size(p.missing)>exp.budget>=exp.size(later.missing)
    assert not p.missing.intersection(later.missing)
    assert not (p.missing|later.missing).intersection(state[7][:-1])
    prefix=[Linear()]
    for obj in exp.candidates[2][base[2]]:
        if obj not in later.missing:
            continue
        path={'RAM':'ram_to_device','SSD':'ssd_to_device'}[exp.objects[obj]['tier']]
        duration=exp._transfer_times[obj]*estimate.get(path,1)/speeds[path]
        if inflight is not None and len(prefix)==1:
            assert obj==inflight[1], 'only the first missing object may be initially active'
            duration=inflight[0]-state[0]
            prefix.append(prefix[-1]+Linear(duration))
            continue
        assert duration>0
        prefix.append(prefix[-1]+(Linear(0,0,duration) if path=='ssd_to_device' else Linear(duration)))
    if inflight is None:
        assert prefix[-1]==later.load
    total_later=prefix[-1]
    r=Linear(0,1,0)
    m=len(prefix)-1
    cells=[]
    for k in range(1,m+2):
        region=box.clip(prefix[k-1]-r,True)
        if region is None:
            continue
        if k<=m:
            region=region.clip(r-prefix[k])
        if region is None:
            continue
        occupied_end=prefix[k] if k<=m else r
        remaining=total_later-prefix[k] if k<=m else Linear()
        pending=[region]
        while pending:
            cell=pending.pop()
            try:
                def maximum(a,b):
                    return a if sign(cell,a-b)>=0 else b
                b0=r+zero.compute
                c0=maximum(b0,total_later)+later.compute
                bd=occupied_end+p.load+p.compute
                cd=occupied_end+p.load+maximum(Linear(p.compute),remaining)+later.compute
                vectors={0:[r,b0,c0],deep:[r,bd,cd]}
            except Split as split:
                pending.extend(split.parts)
            else:
                cells.append((cell,vectors,k))
    assert len(cells)<=9*(m+1)
    upper={c:[None]*3 for c in choices}
    total={c:None for c in choices}
    regret={c:F(0) for c in choices}
    probes=0
    for cell,vectors,k in cells:
        for c in choices:
            diffs=[a-b for a,b in zip(vectors[c],vectors[0])]
            for j,d in enumerate(diffs):
                h=max(d.at(v) for v in cell.vertices)
                upper[c][j]=h if upper[c][j] is None else max(upper[c][j],h)
            h=max(sum(d.at(v) for d in diffs) for v in cell.vertices)
            total[c]=h if total[c] is None else max(total[c],h)
            for b in choices:
                regret[c]=max(regret[c],max(sum((a-bb).at(v) for a,bb in zip(vectors[c],vectors[b]))
                                            for v in cell.vertices))
        points={cell.center()}|{v for v in cell.vertices if cell.contains(v)}
        for point in points:
            for c in choices:
                actual=oracle(exp,state,(0,c,base[2]),estimate,speeds,point)
                expected=[state[0]+t.at(point) for t in vectors[c]]
                assert actual==expected,(k,point,c,actual,expected)
                probes+=1
    full=analyze(exp,record,rectangle=box)
    for row in full:
        c=row['candidate']
        assert F(row['individual_upper'])==max(upper[c])
        assert F(row['total_upper'])==total[c]
        assert F(row['regret'])==regret[c]
    return jsonable(dict(objects=m,prefix=[(p.a,p.x) for p in prefix],cells=len(cells),
                         upper=upper,total=total,regret=regret,analytic_vectors=probes,
                         event_cells=full[0]['cells'],event_vectors=full[0]['probes']))


def run():
    sources=read(ROOT/'docs/evidence/dynamic-lookahead-transfer-order-7820.json')['cases']
    rng=Random(7830)
    model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    cases=[]
    for i in range(64):
        trace=json.loads(json.dumps(sources[i%2]['trace']))
        trace['requests'][1]['arrival_ms']=0
        trace['requests'][2]['arrival_ms']=0
        for path in ('ram_to_device','ssd_to_device'):
            trace['cost'][path]=dict(startup_ms=str(rng.randrange(3)),
                                     bytes_per_ms=str(F(18<<20,rng.randrange(1,17))))
        exp=Capture(trace,model,80)
        exp.run_trigger({},'full')
        rec=exp.records[0]
        record=(1,0,rec[2],(0,0,1152),rec[4],rec[5])
        residual=rec[2][5]-rec[2][0]
        box=Region.rectangle(F(1,4),residual*2,F(1,4),F(7,4))
        result=certificate(exp,record,box)
        result['case']=i
        cases.append(result)
    return dict(seed=7830,configurations=len(cases),cases=cases,
                analytic_cells=sum(c['cells'] for c in cases),
                analytic_vectors=sum(c['analytic_vectors'] for c in cases),
                event_vectors=sum(c['event_vectors'] for c in cases))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--check',type=Path)
    args=parser.parse_args()
    result=run()
    if args.check:
        assert result==read(args.check)
    if args.out:
        with args.out.open('x',encoding='utf-8') as f:
            json.dump(result,f,ensure_ascii=False,indent=2)
    print({k:v for k,v in result.items() if k!='cases'})
