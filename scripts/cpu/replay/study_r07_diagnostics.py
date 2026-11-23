"""Compare the frozen algorithm with counterfactuals from the same decision snapshot."""
import argparse
from collections import Counter
from fractions import Fraction as F
import gzip
import inspect
from itertools import product
import json
from pathlib import Path
import sys

from scripts.cpu.replay.study_r07_arrivals import CONFIG, FIXTURE, MODEL, Experiment, cases, read, response, run


def capture(exp, target):
    source, first = inspect.getsourcelines(Experiment._run)
    guard, = [first+j for j,s in enumerate(source) if 'if approximate == fallback and not' in s]
    data = {}

    def local(frame, event, _):
        if event == 'line' and frame.f_lineno == guard:
            v = frame.f_locals
            if v['i'] == target:
                n = v['arrived']
                assert not exp.per_tier and not v['estimate']
                state = (v['now'], v['head'], n, dict(v['admitted']), set(v['ready']), v['predicted_run'],
                         v['predicted_transfer'], v['active_deps'], set(v['held_reservation']),
                         v['release_time'], dict(v['host_cache']))
                if exp.dynamic:
                    state += (v['copy_until'], v['compute_factor'], v['transfer_factor'])
                data.update(state=state, prefix=list(v['selected'][:target]), speeds=dict(v['current_speeds']),
                            fallback=v['fallback'], approx=v['approximate'], unsettled=bool(v['unsettled']),
                            later=[v['serial_choice'](j) for j in range(target+1,n)])
        return local

    def trace(frame, event, _):
        if event == 'call' and frame.f_code.co_name == 'choose' and frame.f_code.co_filename == inspect.getfile(Experiment):
            return local
        return None

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        result = run(exp,'entry-replay')
    finally:
        sys.settrace(previous)
    return data, result


def trial(exp, data, plan):
    n = len(plan)
    snapshot = data['state'][:2] + (n,) + data['state'][3:]
    result = exp._run('fixed', 0, plan, 0, n, snapshot=snapshot, forecast_speed=data['speeds'], lean_rollout=True)
    return sum(result['completion_ms'])


def minimizer(table, fallback):
    if table[fallback] == min(table.values()):
        return fallback
    return min(table, key=lambda c:(table[c],-c))


def offsets(table):
    origin = min(table.values())
    return {c:float(v-origin) for c,v in table.items()}


def future_pair(exp, data, target, plan, trace, model, budget):
    seen = data['state'][2]
    if seen == len(exp.arrivals):
        return None
    prefix = plan[:target]

    def winners(other):
        per_choice = {}
        for tail in product(*other.eligible[target:]):
            z = other.run('fixed', boundaries=(*prefix, *tail))['response_sum_ms']
            per_choice[tail[0]] = min(per_choice.get(tail[0], z), z)
        return sorted(c for c,z in per_choice.items() if z == min(per_choice.values()))

    before = winners(exp)
    now = data['state'][0]
    for delay in (F(1,100), F(40), F(200)):
        clone = dict(trace, requests=[dict(r, arrival_ms=str(exp.arrivals[j] if j<seen else
                         now+delay+exp.arrivals[j]-exp.arrivals[seen])) for j,r in enumerate(exp.requests)])
        other = Experiment(clone, model, budget)
        snap, actual = capture(other, target)
        if snap != data or actual['boundaries_tokens'][target] != plan[target]:
            continue
        after = winners(other)
        if not set(before)&set(after):
            return dict(delay_after_snapshot_ms=str(delay), original_first=before, replacement_first=after,
                        replacement_arrivals=[str(t) for t in other.arrivals], same_snapshot=True)
    return None


def diagnose(exp, record):
    plan, oracle = record['plan'], record['oracle']['plan']
    target = next((i for i,(a,b) in enumerate(zip(plan,oracle,strict=True)) if a!=b),None)
    if target is None:
        return None
    data,result = capture(exp,target)
    assert list(result['boundaries_tokens']) == plan
    if not data:
        return dict(key=record['key'], target=target, classification='isolated-path/unresolved')
    now, head, n, admitted, ready, running, transfer, active, held, release, host = data['state'][:11]
    def choices(h):
        after = data['later'][:h]
        return {c:trial(exp,data,(*plan[:target],c,*after)) for c in exp.eligible[target]}
    base = data['fallback']
    h4, full = choices(4), choices(n)
    a4,a_all = minimizer(h4,base),minimizer(full,base)
    picked = plan[target]
    gated = base==data['approx'] and not data['unsettled']
    assert picked == (base if gated else a4), (record['key'], picked,base,a4)
    joint_indices = [j for j in range(target+1,n) if len(exp.eligible[j])>1][:2]
    restrictions = {j:exp.eligible[j] for j in joint_indices}
    future_options = [restrictions.get(j,(c,)) for j,c in enumerate(data['later'],target+1)]
    joint, joint_plan = {}, {}
    for c in exp.eligible[target]:
        value, suffix = min((trial(exp,data,(*plan[:target],c,*tail)), tail) for tail in product(*future_options))
        joint[c],joint_plan[c] = value,suffix
    ajoint=minimizer(joint,base)
    actual_means = {c:float(run(exp,'entry-replay',overrides={target:c})['response_mean_ms']) for c in exp.eligible[target]}
    assert abs(actual_means[picked] - record['zero']['entry-replay']['mean']) < 1e-9
    out = dict(key=record['key'], target=target, current=picked, fallback=base, oracle_action=oracle[target],
               pruned_matches_exhaustive=not gated, coarse_return=gated, observed=dict(now=str(now),head=head,arrived=n,
                 running=str(running), transfer=str(transfer), active=list(active), ready=sorted(ready),
                 admitted={i:sorted(s) for i,s in admitted.items()}, held=sorted(held), release=str(release),
                 selected=plan[:target], fixed_followers=data['later'], candidates=list(exp.eligible[target])),
               window4_choice=a4, window4_relative_ms=offsets(h4), all_seen_choice=a_all, all_seen_relative_ms=offsets(full),
               joint_choice=ajoint,joint_relative_ms=offsets(joint),joint_indices=joint_indices,
               joint_tails=joint_plan, closed_loop_forced_mean_ms=actual_means)
    return out, data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('split', choices=('short','holdout','tail')); p.add_argument('data',type=Path); p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(); cfg=read(CONFIG)
    opener = gzip.open if args.data.suffix == '.gz' else open
    with opener(args.data,'rt',encoding='utf-8') as source:
        records={tuple(z['key'].values()):z for z in map(json.loads,source)}
    witnesses = []
    if args.split == 'tail':
        key0 = dict(window=457, rate=16, budget=128, variant='dense')
        exp = next(exp for key, exp in cases(cfg,'holdout') if key == key0)
        recorded = records[tuple(key0.values())]
        a,b = (run(exp,p) for p in ('entry-replay','local-cost'))
        ar,br = response(exp,a),response(exp,b)
        i = max(range(len(ar)),key=lambda j:ar[j]-br[j])
        assert a['boundaries_tokens'][:i+1] == b['boundaries_tokens'][:i+1]
        alternative = [(j,x,y) for j,(x,y) in enumerate(zip(a['boundaries_tokens'],b['boundaries_tokens'],strict=True)) if x!=y]
        checks=[]
        for j,x,y in alternative:
            z = run(exp,'entry-replay',overrides={j:y})
            checks.append(dict(j=j, token_entry=x, token_baseline=y, request_loss_ms=float(response(exp,z)[i]-br[i]),
                               mean_extra_vs_entry_ms=float(z['response_mean_ms']-a['response_mean_ms'])))
        hit = Counter(j for t in recorded['timed'] for j in t['policies']['entry-replay']['late_indices'])
        data = dict(key=key0, request=i, request_id=exp.requests[i]['id'], boundary=a['boundaries_tokens'][i],
                    zero_extra_ms=float(ar[i]-br[i]), queue_extra_ms=float(a['queue_wait_ms'][i]-b['queue_wait_ms'][i]),
                    rounds_gt10=hit[i], checks=checks)
        args.out.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
        print('one causal tail witness')
        return
    for key,exp in cases(cfg,args.split):
        z=records[tuple(key.values())]
        if z['zero']['entry-replay']['mean']-z['oracle']['upper']>1e-6:
            found=diagnose(exp,z)
            witness, data = found if isinstance(found,tuple) else (found,None)
            if witness:
                if args.split=='short' and data:
                    trace = read(FIXTURE)
                    trace['cost']['device_copy'] = {'startup_ms':'0','bytes_per_ms':100_000_000}
                    trace['cost']['ssd_to_device']['bytes_per_ms'] = str(F(trace['cost']['ssd_to_device']['bytes_per_ms'])/key['rate'])
                    trace['requests'] = exp.requests
                    model = read(MODEL)['models'][0]
                    check = Experiment(trace,model,key['budget'])
                    assert check.arrivals == exp.arrivals and check.eligible == exp.eligible
                    witness['future_pair'] = future_pair(exp,data,witness['target'],z['plan'],trace,model,key['budget'])
                witnesses.append(witness)
    args.out.write_text(json.dumps(witnesses,indent=2)+'\n',encoding='utf-8')
    print(args.split, 'diagnosed', len(witnesses),'certified differences')


if __name__=='__main__':
    main()
