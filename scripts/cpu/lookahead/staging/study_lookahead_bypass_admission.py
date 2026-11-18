"""Audit a later small request bypassing an earlier oversized reservation."""

import argparse
import copy
import json
from fractions import Fraction as F
from pathlib import Path

from scripts.cpu.lookahead.robustness.study_lookahead_joint_domain import ROOT, Region, analyze, oracle, read
from scripts.cpu.lookahead.staging.study_lookahead_three_request import Capture, seed_trace
from scripts.cpu.lookahead.staging.study_lookahead_two_request import jsonable


def make_trace():
    trace = seed_trace()
    trace['fixture_id'] = 'later-small-request-bypasses-earlier-large-request'
    trace['purpose'] = 'Synthetic state-order counterexample; not hardware evidence.'
    trace['memory']['prefetch_mib_cases'] = [145]
    trace['scheduling']['future_window_requests'] = 3
    for request in trace['requests'][1:]:
        request['arrival_ms'] = 0
    trace['requests'][2]['candidates'] = [
        candidate for candidate in trace['requests'][2]['candidates']
        if candidate['boundary_tokens'] in (0, 1152)
    ]

    follower = copy.deepcopy(seed_trace()['requests'][2])
    follower['id'] = 'D'
    follower['input_token_runs'] = [{'token_id': 40, 'count': 1152}]
    for candidate in follower['candidates']:
        candidate['load_order'] = [name.replace('C.', 'D.')
                                   for name in candidate['load_order']]
    trace['requests'].append(follower)

    copies = []
    for obj in trace['objects']:
        if obj['id'].startswith('C.'):
            copied = copy.deepcopy(obj)
            copied['id'] = copied['id'].replace('C.', 'D.')
            for run in copied.get('prefix_token_runs', []):
                run['token_id'] = 40
            copies.append(copied)
    trace['objects'].extend(copies)
    return trace


def run():
    trace = make_trace()
    model = read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    exp = Capture(trace, model, 145)
    exp.run_trigger({}, 'full')
    state = exp.records[0][2]
    speeds = {'ram_to_device': F(1), 'ssd_to_device': F(1)}
    plans = {boundary: (0, boundary, 1152, 576)
             for boundary in (0, 576, 1152)}
    outputs = {}
    for boundary, plan in plans.items():
        raw = exp._run('fixed', 0, plan, 100, 4)
        outputs[boundary] = {key: raw[key] for key in
                             ('completion_ms', 'prefetch_reserved_at_ms', 'transfers')}
        assert len(outputs[boundary]['completion_ms']) == 4

    # The reservation set is not prefix-closed: B=0 admits C but not D;
    # B=576 admits B and D while skipping C.
    assert outputs[0]['prefetch_reserved_at_ms'][2] == 0
    assert outputs[0]['prefetch_reserved_at_ms'][3] is None
    assert outputs[576]['prefetch_reserved_at_ms'][1] == 0
    assert outputs[576]['prefetch_reserved_at_ms'][2] is None
    assert outputs[576]['prefetch_reserved_at_ms'][3] == 0
    assert any(x['request'] == 'D' and x['mode'] == 'prefetch'
               for x in outputs[576]['transfers'])
    assert any(x['request'] == 'C' and x['mode'] == 'demand'
               for x in outputs[576]['transfers'])
    assert outputs[0]['completion_ms'] == [F(30), F(38), F(38), F(185, 4)]
    assert outputs[576]['completion_ms'] == [F(30), F(137, 4), F(157, 4), F(87, 2)]
    assert outputs[1152]['completion_ms'] == [F(30), F(30), F(35), F(173, 4)]
    assert sum(outputs[576]['completion_ms']) < sum(outputs[0]['completion_ms'])

    record = (1, 0, state, (0, 0, 1152, 576), {}, speeds)
    rectangle = Region.rectangle(F(5), F(40), F(3, 4), F(5, 4))
    rows = analyze(exp, record, rectangle=rectangle)
    by_candidate = {row['candidate']: row for row in rows}
    assert by_candidate[576]['individual_upper'] == '9/4'
    assert by_candidate[576]['total_upper'] == '-4'
    assert by_candidate[1152]['individual_upper'] == '0'
    assert by_candidate[1152]['total_upper'] == '-41/4'
    assert by_candidate[576]['protected'] == 1152

    return jsonable(dict(
        trace=trace,
        capacity_mib=145,
        outputs=outputs,
        analysis=rows,
        sizes={str(boundary): sum(exp.objects[obj]['bytes']
                                 for obj in set(exp.candidates[1][boundary])-exp.initial)
               for boundary in (0, 576)},
        c_size=sum(exp.objects[obj]['bytes']
                   for obj in set(exp.candidates[2][1152])-exp.initial),
        d_size=sum(exp.objects[obj]['bytes']
                   for obj in set(exp.candidates[3][576])-exp.initial),
        scope=('Four arrived requests, one serial lane, fixed C deep and D shallow, '
               'B chooses zero or shallow; no shared missing objects or active transfer.'),
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args()
    result = run()
    if args.check:
        assert result == read(args.check)
    if args.out:
        with args.out.open('x', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    summary = dict(capacity_mib=result['capacity_mib'],
                   sizes=[result['sizes'], result['c_size'], result['d_size']],
                   completions={k:v['completion_ms'] for k,v in result['outputs'].items()},
                   reservations={k:v['prefetch_reserved_at_ms'] for k,v in result['outputs'].items()},
                   bounds=[{key:row[key] for key in
                            ('candidate','cells','individual_upper','total_upper','regret')}
                           for row in result['analysis']])
    print(json.dumps(summary, ensure_ascii=False))
