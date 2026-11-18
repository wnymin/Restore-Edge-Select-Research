"""Audit snapshot workload intervals against fixed-plan Phase 1 forecasts."""

import argparse
import csv
from fractions import Fraction as F
from pathlib import Path
from random import Random

from scripts.cpu.lookahead.screening.study_lookahead_trigger import Trigger, cases
from scripts.cpu.lookahead.screening.study_lookahead_phase1 import Experiment, ROOT, read, trace_for


def clone(state):
    return (*state[:3], dict(state[3]), set(state[4]), *state[5:8],
            set(state[8]), state[9], dict(state[10]))


class WorkloadBounds(Trigger):
    def interval(self, state, plan, estimate, speeds, *, windowed=True, demand_only=False):
        now, head, n, admitted, ready, running, transfer = state[:7]
        jobs = n - head
        lower = jobs * now
        extra_copy = io = F(0)
        later_objects=set()
        shared_terminals={}
        for j in range(n-1,head-1,-1):
            deps=self.candidates[j][plan[j]]
            shared_terminals[j]=bool(deps) and deps[-1] in later_objects
            later_objects.update(deps)
        if transfer:
            io += jobs * (transfer[0] - now)
        for j in range(head, n):
            self.stats['bound_request_checks'] += 1
            c = plan[j]
            if j == head and running is not None:
                lower += (n-j) * (running-now)
                continue
            deps = self.candidates[j][c]
            lower += (n-j) * self._suffix_times[j][c] * estimate.get('compute', 1)
            shared_terminal = shared_terminals[j]
            if deps and c < self.lengths[j] and (deps[-1] in self.initial or shared_terminal):
                copy = self.path_time('device_copy', self.objects[deps[-1]]['bytes']) * estimate.get('device_copy', 1)
                if deps[-1] in self.initial:
                    lower += (n-j) * copy
                elif shared_terminal:
                    extra_copy += (n-j) * copy
            # Reserved ready objects survive until this request starts. Other
            # ready objects may be discarded at an intervening handoff.
            resident = ready if j in admitted or j == head else self.initial
            missing = set(deps) - self.initial - resident
            # A transfer already in progress is paid above, even if no longer
            # useful to this plan. Do not rely on its result for unreserved jobs.
            if transfer and (j in admitted or j == head):
                missing.discard(transfer[1])
            earliest = head
            if windowed:
                earliest = max(head, j-self.window) if (j in admitted or self.can_prefetch[j][c]) else j
            if demand_only:
                earliest=j
            for obj in missing:
                self.stats['bound_object_checks'] += 1
                path = {'RAM':'ram_to_device', 'SSD':'ssd_to_device'}[self.objects[obj]['tier']]
                io += (n-earliest) * self._transfer_times[obj] * estimate.get(path, 1) / speeds[path]
        return lower, lower + extra_copy + io

    def _run(self, policy, threshold_tokens, boundaries, max_combinations, n, **kwargs):
        state = kwargs.get('snapshot')
        if state is None:
            return Experiment._run(self, policy, threshold_tokens, boundaries, max_combinations, n, **kwargs)
        assert policy == 'fixed' and kwargs['lean_rollout']
        assert not kwargs.get('future_cost') and kwargs.get('completion_cutoff') is None
        if self.cut is None:
            self.cut = 'workload'
            self.best_exact = None
            self.intervals = {}
            self.stats['bound_decisions'] += 1
            for c in self.eligible[self.i]:
                plan = (*boundaries[:self.i], c, *boundaries[self.i+1:])
                self.intervals[c] = self.interval(state, plan, kwargs['estimate'], kwargs['forecast_speed'],
                                                   windowed=self.windowed)
            self.dominant = next((c for c, (_, upper) in self.intervals.items()
                                  if all(c == d or upper < low for d, (low, _) in self.intervals.items())), None)
            self.stats['certified_decisions'] += self.dominant is not None
            self.stats['prefetch_possible_decisions'] += self.last_early[n-1] > self.i
            self.stats['certified_with_prefetch'] += self.dominant is not None and self.last_early[n-1] > self.i
        c = boundaries[self.i]
        low, high = self.intervals[c]
        skip = self.dominant is not None or (self.best_exact is not None and low > self.best_exact)
        actual = None
        if self.audit or not skip:
            actual = Experiment._run(self, policy, threshold_tokens, boundaries, max_combinations, n,
                                     **(kwargs | {'snapshot': clone(state)}))
            score = sum(actual['completion_ms'])
            assert low <= score <= high, (self.i, c, low, score, high, state)
            self.stats['intervals_audited'] += 1
            self.stats['audit_events'] += actual['events_processed']
            if self.audit:
                checks=(self.stats['bound_request_checks'],self.stats['bound_object_checks'])
                _, naive_high=self.interval(state,boundaries,kwargs['estimate'],kwargs['forecast_speed'],
                                             demand_only=True)
                self.stats['bound_request_checks'],self.stats['bound_object_checks']=checks
                self.stats['naive_violations'] += score>naive_high
                self.stats['naive_max_shortfall'] = max(self.stats['naive_max_shortfall'],score-naive_high)
        if skip:
            self.stats['skipped_forecasts'] += 1
            if self.dominant is None:
                surrogate = low
                self.stats['pruned_after_exact'] += 1
            else:
                surrogate = high if c == self.dominant else low
            if self.audit:
                self.stats['saved_events'] += actual['events_processed']
            return dict(completion_ms=[surrogate], events_processed=0)
        self.best_exact = min(self.best_exact, score) if self.best_exact is not None else score
        return actual


def random_cases(model, seed, rounds):
    rng=Random(seed)
    for case in range(rounds):
        mode=rng.choice(('distinct','alias'))
        trace=trace_for(mode,model,free_case=True)
        trace['scheduling']['future_window_requests']=rng.randrange(7)
        trace['cost']['device_copy']=dict(startup_ms='1/10',bytes_per_ms=str(1<<20))
        if mode=='alias' and rng.choice((False,True)):
            for c in trace['requests'][-1]['candidates']:
                c['load_order']=[('C'+key[1:]) if key.startswith('D') else key for key in c['load_order']]
        initial=[o['id'] for o in trace['objects'] if rng.randrange(5)==0]
        trace['scheduling']['initial_device_object_ids']=initial
        now=0
        for r in trace['requests']:
            now+=rng.choice((0,0,1,4,16))
            r['arrival_ms']=now
        trace['io_path_speed_epochs']={p:[[0,str(rng.choice((F(1,2),F(1),F(2))))]]
                                       for p in ('ram_to_device','ssd_to_device')}
        factors={p:rng.choice((F(1,2),F(1),F(3,2)))
                 for p in ('compute','device_copy','ram_to_device','ssd_to_device')}
        yield trace, dict(split='random',source=mode,window=seed,spacing=case,sharing=mode,
                          ssd_scale='',estimate='',budget=rng.choice((0,80,128,248,256,384)),suffix=''), factors


def blocking_witness():
    model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    trace=trace_for('c-only',model,free_case=True)
    for request in trace['requests']:
        request['arrival_ms']=0
    path=trace['cost']['ssd_to_device']
    path['bytes_per_ms']=str(F(path['bytes_per_ms'])/100)
    probe=WorkloadBounds(trace,model,0)
    capacity=F(probe.size(probe.candidates[2][1120]),1<<20)
    exp=WorkloadBounds(trace,model,capacity)
    plan=(0,1152,1120)
    state=(F(0),0,3,{},set(),exp._suffix_times[0][0],None,(),set(),None,{})
    speeds={'ram_to_device':F(1),'ssd_to_device':F(1)}
    result=Experiment._run(exp,'fixed',0,plan,100,3,snapshot=clone(state),estimate={},
                           forecast_speed=speeds,lean_rollout=True)
    score=sum(result['completion_ms'])
    low,naive=exp.interval(state,plan,{},speeds,demand_only=True)
    _,window=exp.interval(state,plan,{},speeds)
    _,global_upper=exp.interval(state,plan,{},speeds,windowed=False)
    assert low<=score<=window<=global_upper and score>naive
    yield dict(split='blocking',budget_mib=str(capacity),plan=str(plan),
               lower=str(low),score=str(score),naive_upper=str(naive),
               window_upper=str(window),global_upper=str(global_upper),
               naive_shortfall_ms=float(score-naive))


def study(args):
    model = read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    source = (random_cases(model,args.seed,args.rounds) if args.split=='random' else
              ((t,l,{'ssd_to_device':F(l['estimate'])}) for t,l in cases(args,model)))
    for trace, labels, estimate in source:
        full = Trigger(trace, model, labels['budget'])
        baseline, _ = full.run_trigger(estimate, 'full')
        triggered=None
        if args.split!='random':
            trigger=Trigger(trace,model,labels['budget'])
            triggered,_=trigger.run_trigger(estimate,'certified')
            assert triggered['boundaries_tokens']==baseline['boundaries_tokens']
            assert triggered['completion_ms']==baseline['completion_ms']
        for windowed in (False, True):
            exp = WorkloadBounds(trace, model, labels['budget'], audit=True)
            exp.windowed = windowed
            out, _ = exp.run_trigger(estimate, 'full')
            assert out['boundaries_tokens'] == baseline['boundaries_tokens']
            assert out['completion_ms'] == baseline['completion_ms']
            assert out['decision_events'] + exp.stats['saved_events'] == baseline['decision_events']
            # Verify that the production path does not require audit forecasts.
            unchecked=WorkloadBounds(trace,model,labels['budget'],audit=False)
            unchecked.windowed=windowed
            fast,_=unchecked.run_trigger(estimate,'full')
            assert fast['boundaries_tokens']==out['boundaries_tokens']
            assert fast['completion_ms']==out['completion_ms']
            assert fast['decision_events']==out['decision_events']
            yield labels | dict(bound='window' if windowed else 'global',
                                decisions=exp.stats['bound_decisions'],
                                certified=exp.stats['certified_decisions'],
                                prefetch_possible=exp.stats['prefetch_possible_decisions'],
                                certified_with_prefetch=exp.stats['certified_with_prefetch'],
                                audited=exp.stats['intervals_audited'],
                                skipped=exp.stats['skipped_forecasts'],
                                pruned_after_exact=exp.stats['pruned_after_exact'],
                                full_events=baseline['decision_events'], events=out['decision_events'],
                                trigger_events=triggered['decision_events'] if triggered else '',
                                saved_events=exp.stats['saved_events'],
                                bound_request_checks=exp.stats['bound_request_checks'],
                                bound_object_checks=exp.stats['bound_object_checks'],
                                naive_violations=exp.stats['naive_violations'],
                                naive_max_shortfall_ms=float(exp.stats['naive_max_shortfall']),
                                response_sum_ms=float(out['response_sum_ms']), changed_vs_full=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=('witness','calibration','holdout','random','blocking'), required=True)
    p.add_argument('--seed',type=int,default=7660)
    p.add_argument('--rounds',type=int,default=200)
    p.add_argument('--out', type=Path, required=True)
    args=p.parse_args()
    with args.out.open('x',newline='',encoding='utf-8') as f:
        writer=None
        count=0
        for row in (blocking_witness() if args.split=='blocking' else study(args)):
            if writer is None:
                writer=csv.DictWriter(f,fieldnames=row)
                writer.writeheader()
            writer.writerow(row)
            count+=1
    print(args.split, count, 'audited rows')


if __name__=='__main__':
    main()
