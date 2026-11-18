"""Equal aggregate costs can require opposite actions with demand-only loading."""

import argparse
from fractions import Fraction as F
import json
from pathlib import Path

from scripts.cpu.lookahead.staging.study_lookahead_three_request import seed_trace
from scripts.cpu.lookahead.robustness.study_lookahead_joint_domain import Capture, Experiment, ROOT, Region, analyze, read
from scripts.cpu.lookahead.staging.study_lookahead_two_request import jsonable


def run():
    model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    cases=[]
    for slow_first in (False,True):
        trace=seed_trace()
        trace['fixture_id']='aggregate-load-order-counterexample'
        trace['scheduling']['initial_device_object_ids']=['C.state2']
        for request in trace['requests'][1:]:
            request['arrival_ms']=25.125
        trace['requests'][1]['candidates']=[c for c in trace['requests'][1]['candidates']
                                            if c['boundary_tokens'] in (0,1152)]
        for key in ('a_ms_per_token_rational','b_ms_per_pair_rational'):
            a,b=trace['cost']['recompute'][key]
            v=F(a,b)*F(15,16)
            trace['cost']['recompute'][key]=[v.numerator,v.denominator]
        for obj in trace['objects']:
            obj['tier']='RAM'
            if obj['id']==('C.kv1' if slow_first else 'C.kv2'):
                obj['tier']='SSD'
        trace['cost']['ram_to_device']=dict(startup_ms=0,bytes_per_ms=str(18<<20))
        trace['cost']['ssd_to_device']=dict(startup_ms=0,bytes_per_ms=str(F(18<<20,5)))
        exp=Capture(trace,model,80)
        exp.run_trigger({},'full')
        rec=exp.records[0]
        assert rec[0]==1 and rec[2][5]-rec[2][0]==3
        state=rec[2]
        record=(*rec[:3],(0,0,1152),*rec[4:])
        record=(record[0],0,*record[2:])
        rows=analyze(exp,record,rectangle=Region.rectangle(F(5,2),F(7,2),F(3,4),F(5,4)))
        plans={b:Experiment._run(exp,'fixed',0,(0,b,1152),100,3) for b in (0,1152)}
        relative={b:[v-state[0] for v in out['completion_ms']] for b,out in plans.items()}
        expected=[F(3),F(467 if slow_first else 515,48),F(515,48)]
        assert relative[1152]==expected and relative[0]==[F(3),F(21,2),F(21,2)]
        delta=sum(relative[1152])-sum(relative[0])
        assert delta==F(-13 if slow_first else 11,24)
        target_deps=set(exp.candidates[1][1152])-exp.initial
        later_deps=set(exp.candidates[2][1152])-exp.initial
        assert exp.size(target_deps)>exp.budget>=exp.size(later_deps)
        aggregate={i:sum(exp._transfer_times[o] for o in deps)
                   for i,deps in ((1,target_deps),(2,later_deps))}
        assert aggregate=={1:F(227,48),2:F(6)}
        cases.append(jsonable(dict(slow_first=slow_first,trace=trace,capacity_mib=80,
                                  aggregate=aggregate,relative_completion=relative,total_deep_delta=delta,
                                  best=1152 if delta<0 else 0,symbolic_rows=rows)))
    assert cases[0]['aggregate']==cases[1]['aggregate']
    return dict(cases=cases,shared_description='Same legal object sizes, missing bytes and aggregate RAM/SSD times; only two equal-sized KV tiers exchange positions.')


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
    print([(c['slow_first'],c['total_deep_delta'],c['best']) for c in result['cases']])
