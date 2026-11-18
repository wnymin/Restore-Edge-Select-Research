"""Long legal object chains and a known initially active first C transfer."""

import argparse
from fractions import Fraction as F
import json
from pathlib import Path
from random import Random

from scripts.cpu.replay.study_boundary_window_phase1 import make_asymmetric
from scripts.cpu.lookahead.staging.study_lookahead_three_request import seed_trace
from scripts.cpu.lookahead.robustness.study_lookahead_joint_domain import Capture, ROOT, Region, read
from scripts.cpu.lookahead.staging.study_lookahead_prefix_certificate import certificate


def run():
    rng=Random(7840)
    model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    cases=[]
    for block in (576,128,64,32):
        for repeat in range(2):
            trace=seed_trace()
            make_asymmetric(trace,model,block=block,short_boundaries=(512 if block!=576 else 576,)*2)
            trace['requests'][1]['candidates']=[c for c in trace['requests'][1]['candidates']
                                                if c['boundary_tokens'] in (0,1152)]
            trace['scheduling']['initial_device_object_ids']=['C.state1152']
            for obj in trace['objects']:
                obj['tier']=rng.choice(('RAM','SSD'))
            for path in ('ram_to_device','ssd_to_device'):
                trace['cost'][path]=dict(startup_ms=str(F(rng.randrange(3),4)),
                                         bytes_per_ms=str(F(18<<20,rng.randrange(1,9))))
            exp=Capture(trace,model,80)
            exp.run_trigger({},'full')
            rec=exp.records[0]
            for active in (False,True):
                state=rec[2]
                if active:
                    obj=next(o for o in exp.candidates[2][1152] if o not in exp.initial)
                    duration=exp._transfer_times[obj]
                    elapsed=min(duration/2,F(1))
                    # Reachable from the all-arrived, fixed deep plan: B cannot
                    # reserve, C reserves and starts its first missing object.
                    state=(elapsed,0,3,{2:set(exp.candidates[2][1152])-exp.initial},
                           set(exp.initial),F(30),(duration,obj),(),set(),None,{})
                record=(1,0,state,(0,0,1152),{},rec[5])
                residual=state[5]-state[0]
                box=Region.rectangle(F(1,4),residual*2,F(1,4),F(7,4))
                result=certificate(exp,record,box)
                result.update(block=block,repeat=repeat,initial_transfer=active)
                cases.append(result)
    return dict(seed=7840,cases=cases,configurations=len(cases),
                max_objects=max(c['objects'] for c in cases),
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
