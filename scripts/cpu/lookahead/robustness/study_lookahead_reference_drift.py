"""KV event audit of locally certified edits with an uncharged fallback change."""

import argparse
from collections import Counter
from copy import deepcopy
import csv
from fractions import Fraction as F
import json
from pathlib import Path

from scripts.cpu.lookahead.robustness.study_lookahead_entry_rollout import EntryRollout
from scripts.cpu.lookahead.screening.study_lookahead_phase1 import Experiment, ROOT, read
from scripts.cpu.lookahead.screening.study_lookahead_workload_bounds import random_cases


class DriftEntry(EntryRollout):
    def audit_drift(self):
        self.epsilon=F(0)
        self.records=[]
        self.current=None
        self.baseline=Experiment._run(self,'local-cost',0,None,100,len(self.requests))
        self.baseplan=list(self.baseline['boundaries_tokens'])
        expected=iter(i for i,c in enumerate(self.eligible) if len(c)>1)

        def select(rows,fallback,later):
            self.target=next(expected)
            assert later==len(self.requests)-self.target-1
            # Current fallback uses observed lane rates. Future serial_choices
            # still use static transfer_time in the original scorer.
            self.baseplan[self.target]=fallback
            self.current=None
            return 0

        actual=self.run('snapshot-screened',estimate={},screen_mode='classified',
                        screen_selector=select,score_candidate=lambda row:0,screen_search='lean')
        assert next(expected,None) is None
        ends=actual['completion_ms']
        actions=[F(0)]*len(ends)
        bridges=[F(0)]*len(ends)
        upper=[F(0)]*len(ends)
        previous=self.baseline['completion_ms']
        log=[]
        for r in self.records:
            head=r['head']
            ref=ends[:head]+r['reference']
            chosen=ends[:head]+r['chosen']
            bridge=[a-b for a,b in zip(ref,previous)]
            action=[a-b for a,b in zip(chosen,ref)]
            assert max(action)<=0
            actions=[a+b for a,b in zip(actions,action)]
            bridges=[a+b for a,b in zip(bridges,bridge)]
            for j,u in enumerate(r['upper'],head):
                upper[j]+=u
            previous=chosen
            log.append({k:r[k] for k in ('head','target','fallback','selected','running')}
                       |{k:list(map(str,v)) for k,v in dict(reference=ref,chosen=chosen,
                                                          bridge=bridge,action=action).items()})
        assert previous==ends
        delta=[a-b for a,b in zip(ends,self.baseline['completion_ms'])]
        assert delta==[a+b for a,b in zip(actions,bridges)]
        assert all(d<=u+b for d,u,b in zip(delta,upper,bridges))
        return dict(baseline=self.baseline['boundaries_tokens'],actual_plan=actual['boundaries_tokens'],
                    baseline_completion=list(map(str,self.baseline['completion_ms'])),
                    actual_completion=list(map(str,ends)),action_sum=list(map(str,actions)),
                    bridge_sum=list(map(str,bridges)),certified_upper=list(map(str,upper)),
                    worst_individual=str(max(delta)),total_delta=str(sum(delta)),decisions_log=log)


def main(args):
    model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    rows=[]
    stats=Counter()
    saved=False
    for case,(trace,labels,_) in enumerate(random_cases(model,args.seed,args.rounds)):
        trace=deepcopy(trace)
        for request in trace['requests']:
            request['arrival_ms']=0
        result=DriftEntry(trace,model,labels['budget']).audit_drift()
        rows.append(dict(seed=args.seed,case=case,capacity=labels['budget'],
                         **{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in result.items()}))
        stats['cases']+=1
        stats['decisions']+=len(result['decisions_log'])
        stats['nonzero_bridge_cases']+=any(F(v) for v in result['bridge_sum'])
        stats['positive_individual_cases']+=F(result['worst_individual'])>0
        stats['positive_total_cases']+=F(result['total_delta'])>0
        if args.witness and not saved and F(result['worst_individual'])>0:
            with args.witness.open('x',encoding='utf-8') as f:
                json.dump(dict(trace=trace,capacity=labels['budget'],**result),f,indent=2)
            saved=True
    with args.out.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with args.summary.open('x',encoding='utf-8') as f:
        json.dump(dict(stats),f,indent=2,sort_keys=True)
    print(json.dumps(dict(stats),sort_keys=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=7752)
    p.add_argument('--rounds',type=int,default=100)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--witness',type=Path)
    main(p.parse_args())
