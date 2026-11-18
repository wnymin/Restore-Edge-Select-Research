"""Audit admission masks that evolve after head requests release reservations."""
import argparse
import copy
import json
from fractions import Fraction as F
from pathlib import Path
from scripts.cpu.lookahead.robustness.study_lookahead_joint_domain import ROOT, read
from random import Random
from scripts.cpu.lookahead.staging.study_lookahead_three_request import Capture, seed_trace
from scripts.cpu.lookahead.staging.study_lookahead_two_request import jsonable


def make_trace():
    t=seed_trace(); t['fixture_id']='dynamic-release-retries-failed-admission'; t['purpose']='Synthetic release-transition audit; not hardware evidence.'
    t['memory']['prefetch_mib_cases']=[145]; t['scheduling']['future_window_requests']=3
    for q in t['requests'][1:]: q['arrival_ms']=0
    # D's KV is initially resident: B shallow + D checkpoint fits, C deep fails; after B starts,
    # C is retried and can coexist with D.
    t['requests'][2]['candidates']=[c for c in t['requests'][2]['candidates'] if c['boundary_tokens'] in (0,1152)]
    d=copy.deepcopy(seed_trace()['requests'][2]); d['id']='D'; d['input_token_runs']=[{'token_id':40,'count':1152}]
    for c in d['candidates']: c['load_order']=[v.replace('C.','D.') for v in c['load_order']]
    t['requests'].append(d)
    for o in list(t['objects']):
        if o['id'].startswith('C.'):
            x=copy.deepcopy(o); x['id']=x['id'].replace('C.','D.')
            for run in x.get('prefix_token_runs',[]): run['token_id']=40
            t['objects'].append(x)
    t['scheduling']['initial_device_object_ids'] = ['D.kv1']
    return t


def admission_epochs(exp, plan):
    """Pure byte-set state machine; no service times are examined."""
    admitted = {}
    epoch = [None]*len(plan)
    masks = []
    for head in range(len(plan)):
        admitted.pop(head,None)
        for request in range(head+1,len(plan)):
            missing = set(exp.candidates[request][plan[request]])-exp.initial
            occupied = set().union(*admitted.values())
            if request not in admitted and missing and exp.size(occupied | missing)<=exp.budget:
                admitted[request]=missing
                epoch[request]=head
        masks.append(tuple(sorted(admitted)))
    return epoch,masks


def epoch_audit():
    rng=Random(7884)
    model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    cases=[]
    for index in range(64):
        trace=make_trace()
        # All candidate compute phases are positive; no zero-stage scan skips.
        for request in trace['requests'][1:]:
            request['input_token_runs'][0]['count']+=128
        if index%2:
            trace['scheduling']['initial_device_object_ids']=[]
        capacity=(80,100,128,145,160,180,220,280)[index%8]
        trace['memory']['prefetch_mib_cases']=[capacity]
        trace['cost']['ram_to_device']['bytes_per_ms']=str(F(18<<20,rng.randrange(1,8)))
        trace['cost']['ssd_to_device']['bytes_per_ms']=str(F(51511296,rng.randrange(1,12)))
        for obj in trace['objects']:
            obj['tier']=rng.choice(('RAM','SSD'))
        exp=Capture(trace,model,capacity)
        plan=(0,(0,576,1152)[index%3],1152,576)
        assert all(exp._suffix_times[j][boundary]>0 for j,boundary in enumerate(plan))
        predicted,masks=admission_epochs(exp,plan)
        raw=exp._run('fixed',0,plan,100,4)
        starts=[arrival+wait+pre for arrival,wait,pre in
                zip(exp.arrivals,raw['queue_wait_ms'],raw['head_to_compute_ms'])]
        expected=[None if head is None else starts[head] for head in predicted]
        assert raw['prefetch_reserved_at_ms']==expected,(index,expected,raw['prefetch_reserved_at_ms'])
        assert len(set(masks))<=len(plan)
        slower=copy.deepcopy(trace)
        for path in ('ram_to_device','ssd_to_device'):
            slower['cost'][path]['bytes_per_ms']=str(F(slower['cost'][path]['bytes_per_ms'])/3)
        twin=Capture(slower,model,capacity)
        twin_predicted,twin_masks=admission_epochs(twin,plan)
        assert twin_predicted==predicted and twin_masks==masks
        second=twin._run('fixed',0,plan,100,4)
        twin_starts=[arrival+wait+pre for arrival,wait,pre in
                     zip(twin.arrivals,second['queue_wait_ms'],second['head_to_compute_ms'])]
        assert second['prefetch_reserved_at_ms']==[None if h is None else twin_starts[h] for h in predicted]
        cases.append(dict(index=index,capacity_mib=capacity,plan=plan,
                          admission_epoch=predicted,masks=masks,
                          compute_starts=starts,reserved=raw['prefetch_reserved_at_ms'],
                          slow_compute_starts=twin_starts,slow_reserved=second['prefetch_reserved_at_ms']))
    return cases


def run():
    t=make_trace(); model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    e=Capture(t,model,145); e.run_trigger({},'full')
    plans=(0,576,1152,576)
    raw=e._run('fixed',0,plans,100,4)
    transfers=raw['transfers']
    # C is first skipped at the initial scan, then admitted after B starts;
    # the second C admission is visible in the reservation timeline.
    reserved=raw['prefetch_reserved_at_ms']
    assert reserved[1]==0 and reserved[2]==30 and reserved[3]==0
    assert any(x['request']=='C' and x['mode']=='prefetch' and x['start_ms']==30 for x in transfers), transfers
    assert raw['completion_ms']==[F(30),F(137,4),F(35),F(157,4)]
    events=[dict(request=x['request'],mode=x['mode'],start=x['start_ms'],end=x['end_ms']) for x in transfers]
    return jsonable(dict(epoch_cases=epoch_audit(),trace=t,capacity_mib=145,plan=plans,completion_ms=raw['completion_ms'],
                         prefetch_reserved_at_ms=reserved,transfer_events=events,
                         release_transition=dict(initial_mask=['B','D'],skipped=['C'],
                                                 after_B_start_mask=['C','D'],
                                                 reason='B reservation releases at compute start; C is retried before its demand load.'),
                         scope='Fixed four-request plan, all arrived, disjoint objects, one serial lane, release_at=start.'))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--check',type=Path);a=p.parse_args();v=run()
 if a.check: assert v==read(a.check)
 if a.out:
  with a.out.open('x',encoding='utf-8') as f: json.dump(v,f,ensure_ascii=False,indent=2)
 print(json.dumps({k:v[k] for k in ('completion_ms','prefetch_reserved_at_ms','release_transition')} | {'epoch_cases':len(v['epoch_cases'])},ensure_ascii=False))
