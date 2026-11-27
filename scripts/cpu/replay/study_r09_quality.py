"""R09 same-snapshot quality and wall-clock bounded, serial closed-loop search."""
import argparse
from collections import Counter
from dataclasses import replace
from fractions import Fraction as F
import gzip
from itertools import permutations
import json
from math import ceil
from pathlib import Path
from time import perf_counter_ns

from restore_edge_select.entry_refinement import EntryRefiner, Refinement
from restore_edge_select.offline_bounds import search
from scripts.cpu.replay.study_r08_refinement import CFG, LOCAL, arrivals, compare
from scripts.cpu.replay.study_r09_slack import BASE, anchor, fixed, starts
from scripts.cpu.replay.study_r09_budget import source_pin
from scripts.validation.run_r08_sealed_holdout import capacity, workloads


class Deadline(Exception):
    pass


class Budgeted:
    """Return frozen on timeout; never pretend an overrun has zero elapsed time."""
    def __init__(self, exp, cfg, budget, guard_ns):
        self.refine = EntryRefiner(exp,cfg)
        self.budget_ns = int(F(budget)*1_000_000)
        self.guard_ns = guard_ns
        self.stats = Counter()
        self.extra_ns = []

    def __call__(self,i,head,arrived,frozen,coarse,running,charged,ready,serial,replay):
        started = perf_counter_ns()
        action = frozen
        if running and head < i:
            self.stats['budget_eligible'] += 1
            end = started + self.budget_ns

            def bounded(fn):
                def step(*args):
                    if perf_counter_ns() + self.guard_ns >= end:
                        raise Deadline
                    value = fn(*args)
                    if perf_counter_ns() >= end:
                        raise Deadline
                    return value
                return step

            try:
                action = self.refine(i,head,arrived,frozen,coarse,running,charged,ready,bounded(serial),bounded(replay))
            except Deadline:
                self.stats['budget_abort'] += 1
            elapsed = perf_counter_ns() - started
            self.extra_ns.append(elapsed)
            if elapsed >= self.budget_ns:
                self.stats['budget_overrun'] += 1
                action = frozen
            self.stats['changed'] += action != frozen
        return action


def variant_set(cfg):
    return {
        'short':replace(cfg,wide=False,joint=False),
        'wide':replace(cfg,joint=False),
        'full':cfg,
        'no-guard':replace(cfg,guard=False),
    }


def quantile(samples, p):
    m = sorted(samples)
    return m[ceil(len(m)*p)-1] if m else None


def metrics(exp,z,local,base):
    cm = compare(exp,z,local,base)
    result = {k:cm[k] for k in ('mean','mean_exact','p95','decision','late','max_local_extra','plan')}
    # Exact request timing stays auditable; does not call these GPU-service times.
    result['completion_ms'] = list(map(str,z['completion_ms']))
    result['compute_start_ms'] = list(map(str,starts(exp,z)))
    result['reservation_ms'] = [str(x) if x is not None else None for x in z['prefetch_reserved_at_ms']]
    result['peak_prefetch_bytes'] = z['peak_prefetch_bytes']
    result['own_prefetch_issued'] = [[t[k] for k in ('request','object','path')]
                                      for t in z['transfers'] if t['mode']=='prefetch']
    result['queued_to_compute_gap_ms'] = float(sum(
        max(F(0),s-max(a,p)) for s,a,p in zip(starts(exp,z),exp.arrivals,[F(0),*z['completion_ms'][:-1]],strict=True)))
    return result


def snapshot_curve(exp, cfg, durations):
    # Derive every candidate on the SAME zero-cost frozen sequence. None may
    # mutate execution state, and timing is around each call, not the observer.
    refs = {k:EntryRefiner(exp,v) for k,v in variant_set(cfg).items()}
    picks = {k:{} for k in refs}
    times = {k:[] for k in refs}

    def observer(*args):
        i,_,_,frozen = args[:4]
        for k,ref in refs.items():
            t0 = perf_counter_ns()
            picks[k][i] = ref(*args)
            times[k].append([i,perf_counter_ns()-t0])
        return frozen

    base = arrivals.run(exp,BASE)
    local = arrivals.run(exp,LOCAL)
    audit = arrivals.run(exp,BASE,extra={'entry_refiner':observer})
    assert arrivals.sequence(audit) == arrivals.sequence(base)
    plans = {}
    report = {}
    base_plan = base['boundaries_tokens']
    for key,choices in picks.items():
        changes = {i:c for i,c in choices.items() if c != base_plan[i]}
        plan = [choices.get(i,c) for i,c in enumerate(base_plan)]
        result = exp.run('fixed',boundaries=plan)
        plans[key] = result
        forced = []
        for i,action in changes.items():
            p = [action if i==j else c for j,c in enumerate(base_plan)]
            z = fixed(exp,p,durations,{})
            refz = fixed(exp,base_plan,durations,{})
            forced.append(dict(index=i,action=action,timed_anchor_fixed_action_gain_ms=str(
                refz['response_mean_ms']-z['response_mean_ms'])))
        report[key] = dict(extra_ns=times[key],changes=changes,result=metrics(exp,result,local,base),
                           isolated_changed_actions=forced,routes=refs[key].stats)
    return local,base,plans,report


def run_arm(exp,cfg,arm,budgets,guard_ns):
    if arm==LOCAL or arm==BASE:
        return arrivals.run(exp,arm,timed=True), {}
    if arm in budgets:
        obj = Budgeted(exp,cfg,budgets[arm],guard_ns)
        res = arrivals.run(exp,BASE,timed=True,extra={'entry_refiner':obj})
        return res, dict(routes=obj.stats,refiner_routes=obj.refine.stats,
                         extra_p95_ns=quantile(obj.extra_ns,.95),extra_max_ns=max(obj.extra_ns,default=0))
    obj = EntryRefiner(exp,variant_set(cfg)[arm])
    return arrivals.run(exp,BASE,timed=True,extra={'entry_refiner':obj}), dict(routes=obj.stats)


def evaluate(exp,key,cfg,budgets,guard_ns,rotation,nodes):
    anchored,_,dur=anchor(exp)
    local,base,plans,snapshot=snapshot_curve(exp,cfg,dur)
    lo,hi,opt,visits=search(exp,[local,base,*plans.values()],nodes,resource_floor=True,dominance=True)
    assert exp.run('fixed',boundaries=opt)['response_sum_ms']==hi
    arms=(LOCAL,BASE,'short','wide','full',*budgets)
    for arm in arms:
        run_arm(exp,cfg,arm,budgets,guard_ns)
    # Latin rotations plus reversal avoid making the first arm always the same.
    order=arms[rotation%len(arms):]+arms[:rotation%len(arms)]
    timed=[]
    for r in range(6):
        p = order[r%len(arms):]+order[:r%len(arms)]
        if (rotation+r)%2:
            p=tuple(reversed(p))
        z={arm:run_arm(exp,cfg,arm,budgets,guard_ns) for arm in p}
        timed.append(dict(order=p,arms={a:dict(metrics(exp,v,z[LOCAL][0],z[BASE][0]),**info) for a,(v,info) in z.items()}))
    n=len(exp.arrivals)
    return dict(key=key,request_ids=anchored['request_ids'],base=metrics(exp,base,local,base),
                local=metrics(exp,local,local,base),snapshot=snapshot,timed=timed,
                offline=dict(lower=str(lo/n),upper=str(hi/n),exact=lo==hi,visits=visits))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',type=Path)
    p.add_argument('budget_lock',type=Path)
    p.add_argument('split',choices=('calibration','holdout'))
    p.add_argument('out',type=Path)
    args=p.parse_args()
    manifest=capacity.read(args.inputs)
    lock=capacity.read(args.budget_lock)
    assert lock['inputs_sha256']==capacity.file_hash(args.inputs)
    assert all(source_pin(capacity.ROOT/k)==pin for k,pin in lock['pins'].items())
    for tr in {a['trace'] for a in manifest[args.split]}:
        assert capacity.file_hash(capacity.TRACE_DIR/tr)==manifest['trace_sha256'][tr]
    budgets={f'{v}ms':F(v) for v in lock['experimental_caps_ms']}
    cfg=Refinement(**capacity.read(CFG)['policy'])
    cases=manifest[args.split]
    with gzip.open(args.out,'xt',encoding='utf-8',newline='\n') as stream:
        for i,(key,exp) in enumerate(workloads(dict(cases=cases,strata=manifest['strata'])),1):
            row=evaluate(exp,key,cfg,budgets,lock['step_guard_ns'],i,lock['oracle_nodes'])
            stream.write(json.dumps(row,separators=(',',':'))+'\n')
            stream.flush()
            print(i,len(cases),key['trace'],key['window'],key['model'],key['stratum'],flush=True)
    assert i==len(cases)


if __name__=='__main__':
    main()
