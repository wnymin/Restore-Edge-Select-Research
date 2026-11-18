"""Joint active compute/transfer residuals; future service costs held fixed."""

import argparse
from fractions import Fraction as F
import json
from pathlib import Path
from random import Random

from scripts.validation.check_joint_domain_evidence import inside
from scripts.cpu.lookahead.robustness.study_lookahead_joint_domain import Capture, ROOT, Region, analyze, oracle, partition, read
from scripts.cpu.lookahead.staging.study_lookahead_two_request import jsonable


def run():
    rng=Random(7850)
    model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    source=read(ROOT/'docs/evidence/dynamic-lookahead-transfer-order-7820.json')['cases']
    cases=[]
    for index in range(48):
        trace=json.loads(json.dumps(source[index%2]['trace']))
        for req in trace['requests'][1:]:
            req['arrival_ms']=0
        for path in ('ram_to_device','ssd_to_device'):
            trace['cost'][path]=dict(startup_ms=str(F(rng.randrange(3),4)),
                                     bytes_per_ms=str(F(18<<20,rng.randrange(1,17))))
        exp=Capture(trace,model,80)
        exp.run_trigger({},'full')
        first=next(o for o in exp.candidates[2][1152] if o not in exp.initial)
        duration=exp._transfer_times[first]
        elapsed=min(F(1),duration/2)
        now=elapsed
        running=F(225,8)
        residual=running-now
        transfer=duration-now
        state=(now,0,3,{2:set(exp.candidates[2][1152])-exp.initial},set(exp.initial),
               running,(duration,first),(),set(),None,{})
        record=(1,0,state,(0,0,1152),{},dict(ram_to_device=F(1),ssd_to_device=F(1)))
        box=Region.rectangle(F(1,4),residual*2,transfer/4,transfer*F(7,4))
        rows=analyze(exp,record,rectangle=box,parameter='active-pair')
        plans={c:(0,c,1152) for c in exp.eligible[1]}
        cells=partition(exp,state,plans,{},record[5],box,'active-pair')
        points={p for region,_ in cells for p in region.vertices}
        points|={(a+(b-a)*F(i,4),c+(d-c)*F(j,4))
                 for a,c in (box.vertices[0],) for b,d in (box.vertices[2],)
                 for i in range(5) for j in range(5)}
        for point in points:
            matches=[vectors for region,vectors in cells if inside(region,point)]
            assert len(matches)==1
            for c,vector in matches[0].items():
                actual=oracle(exp,state,plans[c],{},record[5],point,'active-pair')
                assert actual==[v.at(point) for v in vector]
        cases.append(jsonable(dict(case=index,first=first,nominal_compute_residual=residual,
                                  nominal_transfer_residual=transfer,rows=rows,
                                  partition_grid_vectors=len(points)*len(plans))))
    return dict(seed=7850,configurations=len(cases),cases=cases,
                cells=sum(c['rows'][0]['cells'] for c in cases),
                symbolic_vectors=sum(c['rows'][0]['probes'] for c in cases),
                grid_vectors=sum(c['partition_grid_vectors'] for c in cases),
                corner_gap_candidates=sum(F(r['individual_upper'])>F(r['corner_upper'])
                                          for c in cases for r in c['rows']))


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
