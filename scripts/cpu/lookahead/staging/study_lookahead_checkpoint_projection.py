"""Audit candidate-dependent shared-checkpoint copy and reload costs."""

import argparse
from fractions import Fraction as F
import json
from pathlib import Path
from random import Random

from scripts.cpu.lookahead.staging.study_lookahead_shared_projection import SharedProjection, shared_trace
from scripts.cpu.lookahead.staging.study_lookahead_three_request import critical_points, envelopes
from scripts.cpu.lookahead.robustness.study_lookahead_joint_domain import Capture, ROOT, Region, analyze, oracle, read
from scripts.cpu.lookahead.staging.study_lookahead_two_request import jsonable


def run(seed=7810,rounds=96):
    rng=Random(seed)
    model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    cases=[]
    for index in range(rounds):
        trace=shared_trace()
        trace['objects']=[o for o in trace['objects'] if not o['id'].startswith('C.state')]
        for candidate in trace['requests'][2]['candidates']:
            candidate['load_order']=[o.replace('C.state','B.state') for o in candidate['load_order']]
        for obj in trace['objects']:
            obj['tier']=rng.choice(('RAM','SSD'))
        trace['scheduling']['initial_device_object_ids']=[o['id'] for o in trace['objects'] if rng.randrange(5)==0]
        trace['cost']['device_copy']=dict(startup_ms=str(rng.randrange(3)),bytes_per_ms=str(F(16<<20,rng.randrange(1,8))))
        for path in ('ram_to_device','ssd_to_device'):
            trace['cost'][path]=dict(startup_ms=str(rng.randrange(3)),bytes_per_ms=str(F(16<<20,rng.randrange(1,17))))
        capacity=rng.choice((96,128,160,192))
        fixed=rng.choice((576,1152))
        exp=Capture(trace,model,capacity)
        exp.run_trigger({},'full')
        first=exp.records[0]
        record=(*first[:3],(*first[3][:2],fixed),*first[4:])
        proj=SharedProjection(exp,record,shared_checkpoints=True)
        residual=record[2][5]-record[2][0]
        box=Region.rectangle(residual/4,residual*F(7,4),F(1,4),F(7,4))
        exact=envelopes(proj,box)
        rows=analyze(exp,record,rectangle=box)
        for row in rows:
            c=row['candidate']
            assert F(row['individual_upper'])==exact['individual'][c],(index,c,'individual')
            assert F(row['total_upper'])==exact['total'][c],(index,c,'total')
            assert F(row['regret'])==exact['regret'][c],(index,c,'regret')
            assert row['protected']==exact['protected']
        points=critical_points([d for c in proj.targets for d in proj.hinges(c)],box)
        for point in points:
            for c in proj.targets:
                assert proj.vector(c,point)==oracle(exp,record[2],(0,c,fixed),{},record[5],point),(index,c,point)
        copied=sum(bool(c and c==fixed and exp.candidates[1][c][-1] not in exp.initial and
                        proj.coexist[c] and c<exp.lengths[1]) for c in proj.targets)
        cases.append(jsonable(dict(case=index,capacity=capacity,fixed=fixed,
                                  initial=sorted(exp.initial),coexist=proj.coexist,
                                  computes={c:p.compute for c,p in proj.targets.items()},
                                  remaining={c:(v.a,v.x) for c,v in proj.remaining.items()},
                                  new_checkpoint_copies=copied,exact=exact,
                                  symbolic_cells=rows[0]['cells'],symbolic_vectors=rows[0]['probes'],
                                  critical_vectors=len(points)*len(proj.targets))))
    return dict(seed=seed,configurations=rounds,cases=cases,
                new_checkpoint_copies=sum(c['new_checkpoint_copies'] for c in cases),
                symbolic_cells=sum(c['symbolic_cells'] for c in cases),
                symbolic_vectors=sum(c['symbolic_vectors'] for c in cases),
                critical_vectors=sum(c['critical_vectors'] for c in cases))


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
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},sort_keys=True))
