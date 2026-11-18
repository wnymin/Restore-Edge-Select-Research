"""Separate fixed-reference rollout from active-phase snapshot prediction error.

Future service costs are exact nominal costs (with an SSD sensitivity domain).
Only native active-phase remaining-time forecasts use the phase estimate.
True snapshots are reconstructed after execution strictly for auditing.
"""

import argparse
from collections import Counter
from copy import deepcopy
import csv
from fractions import Fraction as F
import json
from pathlib import Path

from scripts.cpu.lookahead.robustness.study_lookahead_entry_rollout import SnapshotBatch
from scripts.cpu.lookahead.screening.study_lookahead_phase1 import Experiment, ROOT, read, trace_for
from scripts.cpu.lookahead.screening.study_lookahead_workload_bounds import clone, random_cases


PHASES={'exact':{},'compute-under':{'compute':F(3,4)},
        'compute-over':{'compute':F(5,4)},
        'transfer-under':{'ram_to_device':F(3,4),'ssd_to_device':F(3,4)},
        'transfer-over':{'ram_to_device':F(5,4),'ssd_to_device':F(5,4)}}


class SnapshotGap(Experiment):
    def oracle(self,state,plan,speeds):
        return Experiment._run(self,'fixed',0,plan,100,len(plan),snapshot=clone(state),
                               estimate={},forecast_speed=speeds,lean_rollout=True)['completion_ms']

    def audit(self,phase):
        self.records=[]
        self.current=None
        baseline=Experiment._run(self,'local-cost',0,None,100,len(self.requests))
        self.baseplan=tuple(baseline['boundaries_tokens'])
        expected=iter(i for i,c in enumerate(self.eligible) if len(c)>1)

        def select(rows,fallback,later):
            self.target=next(expected)
            assert later==len(self.requests)-self.target-1
            self.native_fallback=fallback
            self.current=None
            return 0

        actual=self.run('snapshot-screened',estimate=PHASES[phase],screen_mode='classified',
                        screen_selector=select,score_candidate=lambda row:0,screen_search='lean')
        assert next(expected,None) is None
        ends=actual['completion_ms']
        ledger=[F(0)]*len(ends)
        predicted_sum=[F(0)]*len(ends)
        gap_sum=[F(0)]*len(ends)
        previous=baseline['completion_ms']
        log=[]
        for r in self.records:
            state=clone(r['state'])
            now,head=state[:2]
            truth=list(state)
            if state[5] is not None:
                truth[5]=ends[head]
            if state[6] is not None:
                obj=state[6][1]
                matches=[t for t in actual['transfers'] if t['object']==obj
                         and t['path'] in ('ram_to_device','ssd_to_device')
                         and t['start_ms']<=now<t['end_ms']]
                assert len(matches)==1,(head,now,obj,matches)
                truth[6]=(matches[0]['end_ms'],obj)
            truth=tuple(truth)
            ref=ends[:head]+self.oracle(truth,r['reference_plan'],r['speeds'])
            chosen=ends[:head]+self.oracle(truth,r['selected_plan'],r['speeds'])
            assert ref==previous,('true reference mismatch',head,r['target'])
            previous=chosen
            pred_ref=ends[:head]+r['predicted_reference']
            pred_chosen=ends[:head]+r['predicted_chosen']
            action=[a-b for a,b in zip(chosen,ref)]
            predicted=[a-b for a,b in zip(pred_chosen,pred_ref)]
            gap=[a-b for a,b in zip(action,predicted)]
            assert max(predicted)<=0
            for j,u in enumerate(r['upper'],head):
                ledger[j]+=u
            predicted_sum=[a+b for a,b in zip(predicted_sum,predicted)]
            gap_sum=[a+b for a,b in zip(gap_sum,gap)]
            log.append(dict(target=r['target'],head=head,selected=r['selected_plan'][r['target']],
                            baseline=self.baseplan[r['target']],native_fallback=r['native_fallback'],
                            running=int(state[5] is not None),inflight=int(state[6] is not None),
                            running_error=str((state[5]-truth[5]) if state[5] is not None else 0),
                            transfer_error=str((state[6][0]-truth[6][0]) if state[6] is not None else 0),
                            upper=list(map(str,[F(0)]*head+list(r['upper']))),
                            predicted_action=list(map(str,predicted)),true_action=list(map(str,action)),
                            paired_model_gap=list(map(str,gap)),reference=list(map(str,ref)),
                            chosen=list(map(str,chosen))))
        assert previous==ends
        delta=[a-b for a,b in zip(ends,baseline['completion_ms'])]
        assert delta==[a+b for a,b in zip(predicted_sum,gap_sum)]
        assert all(d<=u+e for d,u,e in zip(delta,ledger,gap_sum))
        if phase=='exact':
            assert not any(gap_sum) and max(delta)<=0 and sum(delta)<=0
        return dict(baseline=list(self.baseplan),actual_plan=actual['boundaries_tokens'],
                    baseline_completion=list(map(str,baseline['completion_ms'])),
                    actual_completion=list(map(str,ends)),ledger=list(map(str,ledger)),
                    predicted_sum=list(map(str,predicted_sum)),gap_sum=list(map(str,gap_sum)),
                    worst_individual=str(max(delta)),total_delta=str(sum(delta)),
                    uncorrected_ledger_gap=str(max(d-u for d,u in zip(delta,ledger))),
                    decisions_log=log,symbolic_plans=sum(r['plans'] for r in self.records),
                    numeric_plan_points=sum(r['points'] for r in self.records))

    def _run(self,policy,threshold_tokens,boundaries,max_combinations,n,**kwargs):
        state=kwargs.get('snapshot')
        if state is None:
            return Experiment._run(self,policy,threshold_tokens,boundaries,max_combinations,n,**kwargs)
        assert policy=='fixed' and n==state[2]==len(self.requests)
        assert self.target not in state[3]
        if self.current is None:
            batch=SnapshotBatch(self,state,kwargs['forecast_speed'])
            base=tuple(boundaries[j] if j<self.target else self.baseplan[j] for j in range(n))
            assert tuple(boundaries[self.target+1:])==base[self.target+1:]
            best=batch.edit(base,self.target,F(0))
            upper,score=batch.difference(best,base)
            assert max(upper)<=0 and score<=0
            self.records.append(dict(state=clone(state),target=self.target,speeds=kwargs['forecast_speed'],
                                     reference_plan=base,selected_plan=best,upper=upper,
                                     predicted_reference=self.oracle(state,base,kwargs['forecast_speed']),
                                     predicted_chosen=self.oracle(state,best,kwargs['forecast_speed']),
                                     native_fallback=self.native_fallback,
                                     plans=len(batch.cache),points=batch.points))
            self.current=best[self.target]
        # Explicitly enact the certified choice relative to the fixed baseline;
        # native fallback/tie handling must not silently replace that reference.
        return dict(completion_ms=[F(-1) if boundaries[self.target]==self.current else F(0)],
                    events_processed=0)


def constructed_witness(path):
    model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    trace=trace_for('distinct',model,free_case=True)
    trace['requests']=trace['requests'][:2]
    for request in trace['requests']:
        request['arrival_ms']=0
    trace['requests'][1]['candidates']=[c for c in trace['requests'][1]['candidates']
                                      if c['boundary_tokens'] in (0,1152)]
    trace['scheduling']['initial_device_object_ids']=[]
    trace['scheduling']['future_window_requests']=1
    trace['io_path_speed_epochs']={p:[[0,'1']] for p in ('ram_to_device','ssd_to_device')}
    for obj in trace['objects']:
        obj['tier']='RAM'
    exp=Experiment(trace,model,256)
    deps=exp.candidates[1][1152]
    size=sum(exp.objects[k]['bytes'] for k in deps)
    trace['cost']['ram_to_device']=dict(startup_ms='0',bytes_per_ms=str(F(size,39)))
    control_exp=SnapshotGap(trace,model,256)
    control=control_exp.audit('exact')
    result=SnapshotGap(trace,model,256).audit('compute-over')
    assert control['baseline_completion']==control['actual_completion']==['30','38']
    assert result['actual_completion']==['30','39']
    assert result['ledger']==['0','-13/2'] and result['gap_sum']==['0','15/2']
    record=control_exp.records[0]
    for residual in (F(30),F(31),F(75,2)):
        state=list(clone(record['state']))
        assert state[0]==0 and state[1]==0
        state[5]=residual
        ref=control_exp.oracle(tuple(state),(0,0),record['speeds'])
        deep=control_exp.oracle(tuple(state),(0,1152),record['speeds'])
        assert deep[1]-ref[1]==max(F(0),39-residual)-8
    with path.open('x',encoding='utf-8') as f:
        json.dump(dict(trace=trace,capacity=256,phase='compute-over',control=control,**result),
                  f,indent=2)
    print('Constructed witness: forecast gain 13/2, actual harm 1, paired gap 15/2 ms.')


def main(args):
    model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    rows=[]
    stats={phase:Counter() for phase in args.phases}
    saved=set()
    for case,(trace,labels,_) in enumerate(random_cases(model,args.seed,args.rounds)):
        trace=deepcopy(trace)
        for request in trace['requests']:
            request['arrival_ms']=0
        for phase in args.phases:
            result=SnapshotGap(trace,model,labels['budget']).audit(phase)
            rows.append(dict(seed=args.seed,case=case,capacity=labels['budget'],phase=phase,
                             **{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in result.items()}))
            st=stats[phase]
            st['cases']+=1
            st['decisions']+=len(result['decisions_log'])
            st['edits']+=sum(a!=b for a,b in zip(result['baseline'],result['actual_plan']))
            st['native_reference_mismatch_decisions']+=sum(r['native_fallback']!=r['baseline']
                                                         for r in result['decisions_log'])
            st['snapshot_error_decisions']+=sum(F(r['running_error'])!=0 or F(r['transfer_error'])!=0
                                              for r in result['decisions_log'])
            st['positive_individual_cases']+=F(result['worst_individual'])>0
            st['positive_total_cases']+=F(result['total_delta'])>0
            st['uncorrected_ledger_violations']+=F(result['uncorrected_ledger_gap'])>0
            for key in ('symbolic_plans','numeric_plan_points'):
                st[key]+=result[key]
            if args.witness_dir and phase not in saved and F(result['worst_individual'])>0:
                with (args.witness_dir/f'dynamic-lookahead-snapshot-gap-witness-{args.seed}-{phase}.json').open('x',encoding='utf-8') as f:
                    json.dump(dict(trace=trace,capacity=labels['budget'],phase=phase,**result),f,indent=2)
                saved.add(phase)
    with args.out.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with args.summary.open('x',encoding='utf-8') as f:
        json.dump(stats,f,indent=2,sort_keys=True)
    print(json.dumps(stats,sort_keys=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=7760)
    p.add_argument('--rounds',type=int,default=100)
    p.add_argument('--phases',nargs='+',choices=PHASES,default=list(PHASES))
    p.add_argument('--out',type=Path)
    p.add_argument('--summary',type=Path)
    p.add_argument('--witness-dir',type=Path)
    p.add_argument('--constructed-witness',type=Path)
    args=p.parse_args()
    assert args.rounds>0
    if args.constructed_witness:
        constructed_witness(args.constructed_witness)
    else:
        assert args.out is not None and args.summary is not None
        main(args)
