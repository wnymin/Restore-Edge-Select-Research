"""Exhibit exponential capacity-parametric admission masks.

In an all-arrived scan with disjoint reservation sets, admission is ordered
greedy packing. Powers-of-two request sizes yield every subset as capacity
varies. This script also reconstructs those snapshots with the Phase 1 engine.
"""

import argparse
from copy import deepcopy
from fractions import Fraction as F
from itertools import product
from collections import Counter
from math import prod
import json
from pathlib import Path

from scripts.cpu.lookahead.robustness.study_lookahead_joint_domain import ROOT, read
from scripts.cpu.lookahead.screening.study_lookahead_phase1 import Experiment


def make_trace(model, followers):
    trace = deepcopy(read(ROOT/'fixtures/phase1-hand-trace.json'))
    trace.pop('expected_by_prefetch_mib',None)
    trace['fixture_id'] = 'capacity-parametric-binary-admission-family'
    trace['purpose'] = 'Combinatorial construction with legal model payloads; not GPU measurements.'
    trace['scheduling']['future_window_requests'] = followers
    trace['scheduling']['initial_device_object_ids'] = []
    trace['memory']['kv_block_tokens'] = 1 << (followers-1)
    trace['objects'] = []
    trace['requests'] = [dict(id='A',arrival_ms=0,
                              input_token_runs=[dict(token_id=1,count=1)],
                              candidates=[dict(boundary_tokens=0,load_order=[])])]
    unit = model['payload']['attention_bytes_per_token']
    checkpoint = model['payload']['checkpoint_bytes']
    for index in range(followers):
        length = 1 << (followers-index-1)
        token = index+10
        name = f'R{index}'
        kv, state = name+'.kv', name+'.state'
        runs = [dict(token_id=token,count=length)]
        trace['objects'].append(dict(id=kv,kind='attention_kv',tier='RAM',
                                     start_token_exclusive=0,end_token_inclusive=length,
                                     prefix_token_runs=deepcopy(runs),bytes=unit*length))
        trace['objects'].append(dict(id=state,kind='recurrent_and_conv_checkpoint',tier='SSD',
                                     boundary_tokens=length,prefix_token_runs=deepcopy(runs),
                                     bytes=checkpoint))
        # Full-prefix checkpoints are resident and never require a continuation
        # copy; only the unique KV object occupies the variable prefetch pool.
        trace['scheduling']['initial_device_object_ids'].append(state)
        trace['requests'].append(dict(id=name,arrival_ms=0,input_token_runs=runs,
                                     candidates=[dict(boundary_tokens=0,load_order=[]),
                                                 dict(boundary_tokens=length,load_order=[kv,state])]))
    return trace


def greedy_mask(weights, capacity):
    used=mask=0
    for index,weight in enumerate(weights):
        if used+weight <= capacity:
            used+=weight
            mask|=1 << index
    return mask


def structural_checks():
    cases=0
    largest_masks=0
    for n in range(1,7):
        for weights in product((1,2,4),repeat=n):
            counts=Counter(weights)
            type_bound=prod(count+1 for count in counts.values())
            masks={greedy_mask(weights,capacity) for capacity in range(sum(weights)+1)}
            assert len(masks) <= type_bound
            for mask in masks:
                for weight in counts:
                    rejected=False
                    for index,item in enumerate(weights):
                        if item==weight:
                            if mask & (1<<index):
                                assert not rejected
                            else:
                                rejected=True
            if tuple(sorted(weights))==weights:
                assert all(mask & (mask+1)==0 for mask in masks)
                assert len(masks)<=n+1
            largest_masks=max(largest_masks,len(masks))
            cases+=1
    return dict(weight_sequences=cases,max_length=6,alphabet=[1,2,4],
                largest_masks=largest_masks,
                statements=['For d distinct positive sizes, every type has an accepted occurrence prefix; masks <= product_t(n_t+1).',
                            'Nondecreasing positive sizes imply a global accepted prefix; masks <= n+1.'],
                caveat='Disjoint fixed reservation sizes at one initial scan, not arbitrary shared marginal sizes or later admission histories.')


def run(max_followers=8):
    assert 1 <= max_followers <= 10
    model = deepcopy(read(ROOT/'configs/qwen3.5-4b.json')['models'][0])
    rows = []
    for followers in range(1, max_followers+1):
        trace = make_trace(model, followers)
        mib=trace['units']['bytes_per_mib']
        unit=model['payload']['attention_bytes_per_token']
        total_capacity=unit*((1 << followers)-1)
        observed = set()
        for capacity in range(0, total_capacity+1, unit):
            exp = Experiment(trace, model, F(capacity,mib))
            plan = (0, *(request['input_token_runs'][0]['count']
                         for request in trace['requests'][1:]))
            result = exp._run('fixed', 0, plan, 100, followers+1)
            reserved = result['prefetch_reserved_at_ms']
            mask = 0
            for index in range(followers):
                if reserved[index+1] == 0:
                    mask |= 1 << (followers-index-1)
            expected = capacity//unit
            assert mask == expected, (followers, capacity, mask, expected)
            observed.add(mask)
        assert observed == set(range(1 << followers))
        rows.append(dict(followers=followers, capacities_checked=len(observed),
                         distinct_admission_masks=len(observed),
                         theoretical_masks=1 << followers,
                         item_weights_bytes=[unit << i for i in reversed(range(followers))],
                         proof='mask(C) is the binary expansion of floor(C/unit).',
                         capacity_unit_bytes=unit,initial_checkpoint_bytes=followers*model['payload']['checkpoint_bytes'],
                         trace=trace))
    return dict(max_followers=max_followers, rows=rows,structural=structural_checks(),
                scope='All requests arrived, disjoint fixed reservations, one greedy admission scan; capacity is the varying parameter.',
                implication='Explicitly listing all capacity-to-mask outcomes can require 2^n outputs; not a lower bound for implicit representations, time-domain partitions, objective values or optimal boundary algorithms.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-followers', type=int, default=8)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args()
    result = run(args.max_followers)
    if args.check:
        assert result == read(args.check)
    if args.out:
        with args.out.open('x', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key:value for key,value in result.items() if key!='rows'} | {'counts':[row['distinct_admission_masks'] for row in result['rows']]},ensure_ascii=False))
