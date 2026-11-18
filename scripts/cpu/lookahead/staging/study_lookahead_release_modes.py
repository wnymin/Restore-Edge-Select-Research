"""Separate capacity-state transitions from service clocks in fixed Phase 1 plans."""

import argparse
import json
from fractions import Fraction as F
from pathlib import Path
from random import Random

from scripts.cpu.lookahead.staging.study_lookahead_dynamic_release import make_trace
from scripts.cpu.lookahead.robustness.study_lookahead_joint_domain import ROOT, read
from scripts.cpu.lookahead.screening.study_lookahead_phase1 import Experiment
from scripts.cpu.lookahead.staging.study_lookahead_two_request import jsonable


def epochs(exp, plan):
    """Return admission labels and charged request sets in a full-window scan."""
    assert exp.window >= len(plan)-1 and not any(exp.arrivals)
    assert not exp.dynamic and not exp.per_tier and not exp.host_cache_bytes
    assert all(exp._suffix_times[j][b] > 0 for j,b in enumerate(plan))
    missing = [set(exp.candidates[j][b])-exp.initial for j,b in enumerate(plan)]
    queued = set()
    labels = [None]*len(plan)
    states = []

    def scan(head, held, event):
        occupied = held | set().union(*(missing[j] for j in queued))
        for j in range(head+1,len(plan)):
            if j not in queued and missing[j] and exp.size(occupied | missing[j]) <= exp.budget:
                queued.add(j)
                occupied |= missing[j]
                labels[j] = [head,event]
        states.append(dict(head=head,event=event,queued=sorted(queued),held=bool(held)))

    for head,boundary in enumerate(plan):
        if exp.release_at != 'start':
            # At the preceding finish, the old held charge disappears. The
            # incoming head's reservation is charged whether queued or held.
            scan(head,set(),'front')
        held = missing[head] if head in queued else set()
        queued.discard(head)
        if exp.release_at == 'start':
            scan(head,set(),'start')
        elif exp.release_at == 'timer' and held:
            deps = exp.candidates[head][boundary]
            source = exp.initial | set().union(*(missing[j] for j in queued))
            copy = (exp.path_time('device_copy',exp.objects[deps[-1]]['bytes'])
                    if deps and boundary < exp.lengths[head] and deps[-1] in source else 0)
            if exp.release_delay < copy + exp._suffix_times[head][boundary]:
                scan(head,set(),'timer')
    return labels,states


def execute(trace, model, plan, capacity, mode, delay=0):
    trace['scheduling']['prefetch_release'] = mode
    if delay:
        trace['scheduling']['prefetch_hold_ms'] = str(delay)
    exp = Experiment(trace,model,capacity)
    return exp,exp._run('fixed',0,plan,100,len(plan))


def verify(exp, plan, raw):
    labels,states = epochs(exp,plan)
    end = raw['completion_ms']
    start = [end[j]-exp._suffix_times[j][b] for j,b in enumerate(plan)]
    for transfer in raw['transfers']:
        if transfer['path'] == 'device_copy':
            j = raw['request_ids'].index(transfer['request'])
            start[j] = transfer['start_ms']
    expected = []
    for label in labels:
        if label is None:
            expected.append(None)
            continue
        j,event = label
        expected.append({'front': end[j-1] if j else F(0),
                         'start': start[j], 'timer': start[j]+exp.release_delay}[event])
    assert raw['prefetch_reserved_at_ms'] == expected, (exp.release_at,plan,expected,raw)
    return dict(labels=labels,states=states,reserved=expected,completion=end)


def trace_with_sharing(shared):
    trace = make_trace()
    trace['cost']['device_copy'] = dict(startup_ms=1,bytes_per_ms=51511296)
    for req in trace['requests'][1:]:
        req['input_token_runs'][0]['count'] += 128
    if shared:
        # B and C share exact KV and checkpoints at equal boundaries.
        trace['requests'][2]['input_token_runs'][0]['token_id'] = 20
        for candidate in trace['requests'][2]['candidates']:
            candidate['load_order'] = [obj.replace('C.','B.') for obj in candidate['load_order']]
    return trace


def run():
    model = read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    plan = (0,576,1152,576)
    witness = []
    for mode,delay,expected,ce in (
            ('start',0,F(30),F(35)), ('finish',0,None,F(157,4)),
            ('timer',4,F(34),F(39)), ('timer',F(17,4),None,F(157,4)),
            ('timer',5,None,F(157,4))):
        trace = make_trace()
        _,raw = execute(trace,model,plan,145,mode,delay)
        assert raw['prefetch_reserved_at_ms'][2] == expected
        assert raw['completion_ms'][2] == ce
        witness.append(dict(mode=mode,delay=delay,reserved=raw['prefetch_reserved_at_ms'],
                            completion=raw['completion_ms'],transfers=raw['transfers']))

    # Same byte state and 5 ms timer; scaling compute changes event order.
    trace = make_trace()
    a,b = trace['cost']['recompute']['a_ms_per_token_rational']
    trace['cost']['recompute']['a_ms_per_token_rational'] = [2*a,b]
    exp,raw = execute(trace,model,plan,145,'timer',5)
    b_start=raw['completion_ms'][1]-exp._suffix_times[1][576]
    assert raw['prefetch_reserved_at_ms'][2] == b_start+5 < raw['completion_ms'][1]
    slow_compute=dict(trace=trace,reserved=raw['prefetch_reserved_at_ms'],completion=raw['completion_ms'])

    rng = Random(7885)
    cases=[]
    for index in range(72):
        shared=bool((index//3)%2)
        mode=('start','finish','timer')[index%3]
        delay=rng.choice((1,4,6,12)) if mode=='timer' else 0
        capacity=rng.choice((80,100,128,145,160,180,220,280))
        c=1152
        b=rng.choice((0,576,1152))
        fixed=(0,b,c,576)
        results=[]
        for speed in (1,3):
            trace=trace_with_sharing(shared)
            for path in ('ram_to_device','ssd_to_device'):
                trace['cost'][path]['bytes_per_ms']=str(F(trace['cost'][path]['bytes_per_ms'])/speed)
            exp,raw=execute(trace,model,fixed,capacity,mode,delay)
            results.append(verify(exp,fixed,raw))
        assert results[0]['labels']==results[1]['labels'] and results[0]['states']==results[1]['states']
        cases.append(dict(shared=shared,mode=mode,delay=delay,capacity=capacity,plan=fixed,results=results))
    return jsonable(dict(seed=7885,trace=make_trace(),plan=plan,capacity=145,witness=witness,
                         slow_compute=slow_compute,cases=cases))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--check',type=Path)
    args=parser.parse_args()
    result=run()
    if args.check:
        assert result==read(args.check)
    if args.out:
        with args.out.open('x',encoding='utf-8') as handle:
            json.dump(result,handle,ensure_ascii=False,indent=2)
    print(f"Verified {len(result['cases'])} paired configurations and {len(result['witness'])} release cases.")
