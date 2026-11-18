"""Audit time-shift coalescence of fixed FIFO candidate forecasts."""

import argparse
import csv
from fractions import Fraction as F
from pathlib import Path

from scripts.cpu.lookahead.screening.study_lookahead_trigger import Trigger, cases
from scripts.cpu.lookahead.screening.study_lookahead_workload_bounds import clone, random_cases
from scripts.cpu.lookahead.screening.study_lookahead_phase1 import Experiment, ROOT, read, trace_for


class Coalescence(Trigger):
    def signature(self, state, plan, *, coarse=False):
        now, head, n, admitted, ready, running, transfer, active, held, release, host = state
        assert running is None and not held and release is None and not host
        future=self.future_objects[head]
        reservations=tuple((j,frozenset(deps)) for j,deps in sorted(admitted.items()))
        inflight=None if transfer is None else (transfer[0]-now, transfer[1])
        if coarse:
            return (head,tuple((j,self.size(deps)) for j,deps in sorted(admitted.items())),
                    len(ready & future),None if inflight is None else inflight[0])
        if self.project:
            ready=ready & future
            if inflight is not None and inflight[1] not in future:
                inflight=(inflight[0],None)
        return head,reservations,frozenset(ready),inflight

    def _run(self, policy, threshold_tokens, boundaries, max_combinations, n, **kwargs):
        state=kwargs.get('snapshot')
        if state is None:
            return Experiment._run(self,policy,threshold_tokens,boundaries,max_combinations,n,**kwargs)
        assert policy=='fixed' and kwargs['lean_rollout']
        assert not kwargs.get('future_cost') and kwargs.get('completion_cutoff') is None
        start_head=state[1]
        full=None
        if self.audit:
            full=Experiment._run(self,policy,threshold_tokens,boundaries,max_combinations,n,
                                 **(kwargs | {'snapshot':clone(state)}))
            self.stats['audit_forecasts']+=1
            self.stats['audit_full_events']+=full['events_processed']
        if self.cut is None:
            self.cut='coalescence'
            self.cache={}
            self.coarse_cache={}
            self.past_ends=[]
            self.naive_scores={}
            self.exact_scores={}
            self.static_cut=n
            while self.static_cut>self.i+1 and self.static_cut-1 not in state[3] and not self.can_prefetch[self.static_cut-1][boundaries[self.static_cut-1]]:
                self.static_cut-=1
            self.future_objects={n:set()}
            for j in range(n-1,self.i,-1):
                self.future_objects[j]=self.future_objects[j+1] | set(self.candidates[j][boundaries[j]])
            self.stats['decisions']+=1
        ends=[]
        visited=[]
        events=0
        current=clone(state)
        while current[1]<n:
            stop=max(self.i+1,current[1]+1)
            part=Experiment._run(self,policy,threshold_tokens,boundaries,max_combinations,n,
                                 **(kwargs | {'snapshot':clone(current),'pause_at_head':stop}))
            events+=part['events_processed']
            ends.extend(part['completion_ms'])
            self.stats['chunks']+=1
            if 'snapshot' not in part:
                break
            current=part['snapshot']
            k=current[1]
            self.stats['boundary_checks']+=1
            key=self.signature(current,boundaries)
            coarse=self.signature(current,boundaries,coarse=True)
            visited.append((key,coarse,current[0],len(ends)))
            if self.audit and coarse in self.coarse_cache:
                predicted=[current[0]+d for d in self.coarse_cache[coarse]]
                true=full['completion_ms'][len(ends):]
                error=sum(predicted)-sum(true)
                self.stats['coarse_matches']+=1
                self.stats['coarse_wrong']+=predicted!=true
                self.stats['coarse_max_error']=max(self.stats['coarse_max_error'],abs(error))
            if key in self.cache:
                ends.extend(current[0]+d for d in self.cache[key])
                self.stats['merged_forecasts']+=1
                self.stats['reused_completions']+=n-k
                self.stats['merges_with_reservations']+=bool(current[3])
                self.stats['merges_with_inflight']+=current[6] is not None
                self.stats['merges_before_static_cut']+=k<self.static_cut
                break
        if self.audit:
            assert ends==full['completion_ms'],(self.i,boundaries,ends,full)
            self.stats['checked_forecasts']+=1
        for key,coarse,now,done in visited:
            relative=tuple(t-now for t in ends[done:])
            if key in self.cache:
                assert self.cache[key]==relative
            self.cache[key]=relative
            self.coarse_cache.setdefault(coarse,relative)
        prefix=self.i+1-start_head
        for previous in self.past_ends if prefix<len(ends) else ():
            naive=sum(ends[:prefix])-sum(previous[:prefix])
            naive+=(len(ends)-prefix)*(ends[prefix-1]-previous[prefix-1])
            true=sum(ends)-sum(previous)
            self.stats['naive_pairs']+=1
            self.stats['naive_wrong']+=naive!=true
            self.stats['naive_sign_wrong']+=(naive>0)!=(true>0)
            self.stats['naive_max_error']=max(self.stats['naive_max_error'],abs(naive-true))
        self.past_ends.append(ends)
        c=boundaries[self.i]
        self.naive_scores[c]=sum(ends[:prefix])+(len(ends)-prefix)*ends[prefix-1]
        self.exact_scores[c]=sum(ends)
        if len(self.exact_scores)==len(self.eligible[self.i]):
            def winner(scores):
                best=min(scores,key=lambda c:(scores[c],-c))
                return self.fallback if scores[best]>=scores[self.fallback] else best
            self.stats['naive_decision_wrong']+=winner(self.naive_scores)!=winner(self.exact_scores)
        self.stats['evaluated_events']+=events
        return dict(completion_ms=ends,events_processed=events)


def study(args):
    model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    source=(random_cases(model,args.seed,args.rounds) if args.split=='random' else
            ((t,l,{'ssd_to_device':F(l['estimate'])}) for t,l in cases(args,model)))
    for trace,labels,estimate in source:
        reference=Trigger(trace,model,labels['budget'])
        full,_=reference.run_trigger(estimate,'full')
        for project in (False,True):
            exp=Coalescence(trace,model,labels['budget'],audit=True)
            exp.project=project
            out,_=exp.run_trigger(estimate,'full')
            assert out['boundaries_tokens']==full['boundaries_tokens']
            assert out['completion_ms']==full['completion_ms']
            assert exp.stats['audit_full_events']==full['decision_events']
            fast=Coalescence(trace,model,labels['budget'],audit=False)
            fast.project=project
            check,_=fast.run_trigger(estimate,'full')
            assert check['boundaries_tokens']==out['boundaries_tokens']
            assert check['completion_ms']==out['completion_ms']
            assert check['decision_events']==out['decision_events']
            counts={k:exp.stats[k] for k in ('decisions','checked_forecasts','chunks','boundary_checks',
                    'merged_forecasts','reused_completions','merges_with_reservations','merges_with_inflight',
                    'merges_before_static_cut','coarse_matches','coarse_wrong','naive_pairs','naive_wrong',
                    'naive_sign_wrong','naive_decision_wrong')}
            yield labels | dict(signature='relevant' if project else 'exact',**counts,
                                 coarse_max_error_ms=float(exp.stats['coarse_max_error']),
                                 naive_max_error_ms=float(exp.stats['naive_max_error']),
                                 full_events=full['decision_events'],events=out['decision_events'],
                                 changed_vs_full=0)


def state_witnesses():
    model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    trace=trace_for('distinct',model,free_case=True)
    for request in trace['requests']:
        request['arrival_ms']=0
    path=trace['cost']['ssd_to_device']
    path['bytes_per_ms']=str(F(path['bytes_per_ms'])/100)
    exp=Coalescence(trace,model,256)
    exp.project=True
    plan=(0,1152,1152,1152)
    exp.future_objects={j:set().union(*(set(exp.candidates[k][plan[k]]) for k in range(j,4)))
                        for j in range(1,4)}
    deps=exp.candidates[2][1152]
    base=(F(0),1,4,{2:set(deps)},set(),None,None,(),set(),None,{})
    speeds={'ram_to_device':F(1),'ssd_to_device':F(1)}
    def run(state,stop=None):
        return Experiment._run(exp,'fixed',0,plan,100,4,snapshot=clone(state),estimate={},
                               forecast_speed=speeds,lean_rollout=True,pause_at_head=stop)
    ready_a=(*base[:4],{deps[0]},*base[5:])
    ready_b=(*base[:4],{deps[-1]},*base[5:])
    flight_a=(*base[:6],(F(100),deps[0]),*base[7:])
    flight_b=(*base[:6],(F(100),deps[-1]),*base[7:])
    shifted=(F(7),*flight_a[1:6],(F(107),deps[0]),*flight_a[7:])
    stale=(*base[:4],{'B.state896'},*base[5:])
    residual=(*base[:6],(F(150),deps[0]),*base[7:])
    pairs=(('ready_identity',ready_a,ready_b,False),
           ('inflight_identity',flight_a,flight_b,False),
           ('time_shift_inflight',flight_a,shifted,True),
           ('irrelevant_ready',base,stale,True),
           ('inflight_remaining',flight_a,residual,False))
    for name,a,b,should_merge in pairs:
        left,right=run(a),run(b)
        dt=b[0]-a[0]
        same=exp.signature(a,plan)==exp.signature(b,plan)
        assert same==should_merge
        error=sum(right['completion_ms'])-sum(left['completion_ms'])-3*dt
        if should_merge:
            assert right['completion_ms']==[c+dt for c in left['completion_ms']]
        else:
            assert error!=0
        for state,full in ((a,left),(b,right)):
            for stop in (2,3):
                part=run(state,stop)
                continuation=run(part['snapshot'])
                assert part['completion_ms']+continuation['completion_ms']==full['completion_ms']
        yield dict(case=name,signature_equal=int(same),
                   coarse_equal=int(exp.signature(a,plan,coarse=True)==exp.signature(b,plan,coarse=True)),
                   exact_ready_equal=int(a[4]==b[4]),shift=str(dt),
                   left_sum=str(sum(left['completion_ms'])),right_sum=str(sum(right['completion_ms'])),
                   translation_error=str(error),pause_resume_checks=4)
    # The final boundary needs no resumable snapshot. Invalid boundaries and
    # not-yet-arrived requests must fail instead of silently changing semantics.
    assert run(base,4)['completion_ms']==run(base)['completion_ms']
    for stop in (1,5):
        try:
            run(base,stop)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid pause boundary accepted')
    late=(F(-1),*base[1:])
    try:
        run(late,2)
    except ValueError:
        pass
    else:
        raise AssertionError('Future arrivals accepted by stationary pause')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=('witness','calibration','holdout','random','states'),required=True)
    p.add_argument('--seed',type=int,default=7670)
    p.add_argument('--rounds',type=int,default=200)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    with args.out.open('x',newline='',encoding='utf-8') as f:
        writer=None
        count=0
        for row in (state_witnesses() if args.split=='states' else study(args)):
            if writer is None:
                writer=csv.DictWriter(f,fieldnames=row)
                writer.writeheader()
            writer.writerow(row)
            count+=1
    print(args.split,count,'audited rows')


if __name__=='__main__':
    main()
