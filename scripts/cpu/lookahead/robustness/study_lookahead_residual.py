"""Exact parametric Phase 1 analysis of one residual transfer duration."""

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction as F
from pathlib import Path
from random import Random

from scripts.cpu.lookahead.screening.study_lookahead_workload_bounds import clone, random_cases
from scripts.cpu.lookahead.screening.study_lookahead_phase1 import Experiment, ROOT, read


@dataclass(frozen=True)
class Affine:
    a: F=F(0)
    b: F=F(0)

    def at(self,x):
        return self.a+self.b*x

    def __add__(self,other):
        if not isinstance(other,Affine):
            other=Affine(F(other))
        return Affine(self.a+other.a,self.b+other.b)

    __radd__=__add__

    def __sub__(self,other):
        if not isinstance(other,Affine):
            other=Affine(F(other))
        return Affine(self.a-other.a,self.b-other.b)


class Domain:
    def __init__(self,lo,hi,left=True,right=True):
        self.lo,self.hi,self.left,self.right=lo,hi,left,right

    def lower(self,x,closed):
        if x>self.lo:
            self.lo,self.left=x,closed
        elif x==self.lo:
            self.left &= closed

    def upper(self,x,closed):
        if x<self.hi:
            self.hi,self.right=x,closed
        elif x==self.hi:
            self.right &= closed

    def nonempty(self):
        return self.lo<self.hi or self.lo==self.hi and self.left and self.right

    def contains(self,x):
        return (x>self.lo or x==self.lo and self.left) and (x<self.hi or x==self.hi and self.right)


def symbolic(exp,state,plan,estimate,speeds,x,domain,parameter='residual'):
    """Path constraints certify a cell, rather than infer it from samples."""
    now,head,n,admitted,ready,running,transfer=clone(state)[:7]
    assert parameter in ('residual','ssd-scale') and x>0
    assert parameter!='residual' or transfer is not None
    assert not state[8] and state[9] is None and not state[10]
    now=Affine(now)
    if transfer is not None:
        transfer=((now+Affine(0,1)) if parameter=='residual' else Affine(transfer[0]),transfer[1])
    running=None if running is None else Affine(running)
    active=state[7]
    resume=True
    ends=[]
    events=[]
    def compare(a,b,kind):
        d=a-b
        value=d.at(x)
        result=(value==0) if kind=='eq' else (value<0)
        if d.b:
            root=-d.a/d.b
            if value==0:
                domain.lower(root,True)
                domain.upper(root,True)
            elif (value>0)==(d.b>0):
                domain.lower(root,False)
            else:
                domain.upper(root,False)
        return result
    def occupied():
        return set().union(*admitted.values())-exp.initial
    while head<n:
        if not resume and transfer is not None and compare(transfer[0],now,'eq'):
            ready.add(transfer[1])
            events.append(('ready',head,transfer[1]))
            transfer=None
        if not resume and running is not None and compare(running,now,'eq'):
            ends.append(now)
            events.append(('finish',head))
            head+=1
            running=None
            if head==n:
                break
        deps=exp.candidates[head][plan[head]]
        if running is None and ready.issuperset(deps):
            admitted.pop(head,None)
            shared=bool(deps) and (deps[-1] in exp.initial or deps[-1] in occupied())
            copy=(exp.path_time('device_copy',exp.objects[deps[-1]]['bytes'])*estimate.get('device_copy',1)
                  if shared and plan[head]<exp.lengths[head] else 0)
            active=deps
            running=now+copy+exp._suffix_times[head][plan[head]]*estimate.get('compute',1)
            ready.intersection_update(occupied()|exp.initial)
            events.append(('start',head,shared))
            if compare(running,now,'eq'):
                resume=False
                continue
        for j in range(head+1,min(n,head+1+exp.window)):
            future=set(exp.candidates[j][plan[j]])-exp.initial
            if j not in admitted and future and exp.size(occupied()|future)<=exp.budget:
                admitted[j]=future
                if running is not None:
                    ready.update(future.intersection(active[:-1])-ready)
                events.append(('admit',head,j))
        if transfer is None:
            order=[] if running is not None else list(deps)
            for j in sorted(admitted):
                order.extend(exp.candidates[j][plan[j]])
            for obj in order:
                if obj not in ready:
                    path={'RAM':'ram_to_device','SSD':'ssd_to_device'}[exp.objects[obj]['tier']]
                    duration=exp._transfer_times[obj]*estimate.get(path,1)/speeds[path]
                    if parameter=='ssd-scale' and path=='ssd_to_device':
                        duration=Affine(0,duration)
                    transfer=(now+duration,obj)
                    events.append(('transfer',head,obj))
                    break
        times=[t for t in (running,transfer[0] if transfer else None) if t is not None]
        assert times
        now=times[0]
        for t in times[1:]:
            if compare(t,now,'lt'):
                now=t
        resume=False
        assert len(events)<10000
    assert domain.nonempty() and domain.contains(x)
    assert all(t.b>=0 for t in ends)
    if parameter=='residual':
        assert all(t.b in (0,1) for t in ends)
    return ends,tuple(events)


def partition(exp,state,plan,estimate,speeds,lo,hi,parameter='residual'):
    pending=[Domain(lo,hi)]
    cells=[]
    while pending:
        region=pending.pop()
        x=(region.lo+region.hi)/2
        cell=Domain(region.lo,region.hi,region.left,region.right)
        ends,events=symbolic(exp,state,plan,estimate,speeds,x,cell,parameter)
        cells.append((cell,ends,events))
        left=Domain(region.lo,cell.lo,region.left,not cell.left)
        right=Domain(cell.hi,region.hi,not cell.right,region.right)
        pending.extend(p for p in (left,right) if p.nonempty())
        assert len(cells)<1000
    cells.sort(key=lambda c:(c[0].lo,c[0].hi))
    assert cells[0][0].lo==lo and cells[-1][0].hi==hi
    for a,b in zip(cells,cells[1:]):
        assert a[0].hi==b[0].lo
        assert a[0].right!=b[0].left
    return cells


def snapshots(exp,plan,estimate,speeds):
    n=len(plan)
    state=(max(exp.arrivals),0,n,{},set(exp.initial),None,None,(),set(),None,{})
    for k in range(1,n):
        part=Experiment._run(exp,'fixed',0,plan,100,n,snapshot=clone(state),estimate=estimate,
                             forecast_speed=speeds,lean_rollout=True,pause_at_head=k)
        state=part['snapshot']
        if state[6] is not None and state[6][0]>state[0]:
            yield state


def study(args):
    model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    rng=Random(args.seed+1)
    found=0
    exported=False
    for trace,labels,estimate in random_cases(model,args.seed,args.rounds):
        if args.extended:
            for j in range(rng.randrange(1,7)):
                request=deepcopy(rng.choice(trace['requests'][1:]))
                request['id']=f'X{j}'
                distinct=rng.randrange(3)==0
                if distinct:
                    request['input_token_runs'][0]['token_id']=1000+j
                objects={o['id']:o for o in trace['objects']}
                mapping={}
                for c in request['candidates']:
                    for obj in c['load_order']:
                        if obj in mapping:
                            continue
                        if not distinct and rng.randrange(3):
                            mapping[obj]=obj
                            continue
                        copied=deepcopy(objects[obj])
                        copied['id']=f'X{j}:{obj}'
                        if distinct:
                            copied['prefix_token_runs'][0]['token_id']=1000+j
                        trace['objects'].append(copied)
                        mapping[obj]=copied['id']
                for c in request['candidates']:
                    c['load_order']=[mapping[o] for o in c['load_order']]
                trace['requests'].append(request)
            trace['scheduling']['future_window_requests']=rng.randrange(1,9)
            for r in trace['requests']:
                r['arrival_ms']=0
            trace['cost']['ram_to_device']['startup_ms']=str(rng.randrange(4))
            trace['cost']['ssd_to_device']['startup_ms']=str(rng.randrange(9))
        exp=Experiment(trace,model,labels['budget'])
        plan=tuple(rng.choice(choices) for choices in exp.eligible)
        speeds={p:values[1][0] for p,values in exp.io_epochs.items()}
        for state in snapshots(exp,plan,estimate,speeds):
            found+=1
            base=state[6][0]-state[0]
            lo,hi=F(1,1000),max(F(64),base*4)
            cells=partition(exp,state,plan,estimate,speeds,lo,hi)
            if args.witness is not None and not exported:
                for previous,current in zip(cells,cells[1:]):
                    boundary=previous[0].hi
                    jump=sum(t.at(boundary) for t in current[1])-sum(t.at(boundary) for t in previous[1])
                    if jump<0:
                        payload=dict(seed=args.seed,case=labels['spacing'],trace=trace,capacity_mib=labels['budget'],
                                     plan=plan,estimate={k:str(v) for k,v in estimate.items()},
                                     speeds={k:str(v) for k,v in speeds.items()},boundary=str(boundary),
                                     snapshot=dict(now=str(state[0]),head=state[1],n=state[2],
                                         admitted={str(j):sorted(d) for j,d in state[3].items()},ready=sorted(state[4]),
                                         transfer=[str(state[6][0]),state[6][1]],active=list(state[7])),
                                     jump=str(jump),previous_events=previous[2],next_events=current[2])
                        with args.witness.open('x',encoding='utf-8') as f:
                            json.dump(payload,f,ensure_ascii=False,indent=2)
                        exported=True
                        break
            jobs=state[2]-state[1]
            for index,(domain,ends,events) in enumerate(cells):
                points={(domain.lo+domain.hi)/2,
                        (3*domain.lo+domain.hi)/4,(domain.lo+3*domain.hi)/4}
                if domain.left:
                    points.add(domain.lo)
                if domain.right:
                    points.add(domain.hi)
                for r in points:
                    changed=(*state[:6],(state[0]+r,state[6][1]),*state[7:])
                    result=Experiment._run(exp,'fixed',0,plan,100,len(plan),snapshot=clone(changed),
                                           estimate=estimate,forecast_speed=speeds,lean_rollout=True)
                    assert result['completion_ms']==[t.at(r) for t in ends],(args.seed,labels,r,domain.__dict__)
                score=sum(ends,Affine())
                assert 0<=score.b<=jobs
                yield dict(seed=args.seed,extended=int(args.extended),case=labels['spacing'],snapshot=found,head=state[1],
                           plan=str(plan),budget=labels['budget'],inflight=state[6][1],base_residual=str(base),
                           cell=index,cells=len(cells),lo=str(domain.lo),hi=str(domain.hi),
                           left=int(domain.left),right=int(domain.right),
                           intercept=str(score.a),slope=str(score.b),jobs=jobs,
                           events=len(events),audited_points=len(points),
                           fingerprint=hashlib.sha256(repr(events).encode()).hexdigest())
    print('snapshots',found)


def replay_witness(path):
    payload=json.loads(path.read_text(encoding='utf-8'))
    model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    exp=Experiment(payload['trace'],model,payload['capacity_mib'])
    s=payload['snapshot']
    state=(F(s['now']),s['head'],s['n'],{int(j):set(d) for j,d in s['admitted'].items()},
           set(s['ready']),None,(F(s['transfer'][0]),s['transfer'][1]),tuple(s['active']),set(),None,{})
    estimate={k:F(v) for k,v in payload['estimate'].items()}
    speeds={k:F(v) for k,v in payload['speeds'].items()}
    assert any(s==state for s in snapshots(exp,tuple(payload['plan']),estimate,speeds)), \
        'Archived state must be reachable from its fixed-plan saturated start'
    boundary=F(payload['boundary'])
    def score(r):
        changed=(*state[:6],(state[0]+r,state[6][1]),*state[7:])
        out=Experiment._run(exp,'fixed',0,payload['plan'],100,s['n'],snapshot=clone(changed),
                            estimate=estimate,forecast_speed=speeds,lean_rollout=True)
        return sum(out['completion_ms'])
    at_boundary=score(boundary)
    for power in (2,3,6,9):
        epsilon=F(1,10**power)
        left,right=score(boundary-epsilon),score(boundary+epsilon)
        change=right-left
        jobs=s['n']-s['head']
        assert change<0 and abs(change)>jobs*2*epsilon
        yield dict(boundary=str(boundary),epsilon=str(epsilon),left=str(left),
                   at_boundary=str(at_boundary),right=str(right),change=str(change),
                   residual_change=str(2*epsilon),jobs=jobs,
                   amplification=str(abs(change)/(2*epsilon)))


def summarize(path):
    groups=defaultdict(list)
    for row in csv.DictReader(path.open(encoding='utf-8')):
        groups[row['snapshot']].append(row)
    jumps=[]
    affected=set()
    for key,rows in groups.items():
        for left,right in zip(rows,rows[1:]):
            x=F(left['hi'])
            jump=F(right['intercept'])+F(right['slope'])*x-F(left['intercept'])-F(left['slope'])*x
            if jump:
                affected.add(key)
                jumps.append(dict(snapshot=key,boundary=str(x),jump=str(jump)))
    rows=[r for rs in groups.values() for r in rs]
    return dict(snapshots=len(groups),cells=len(rows),
                audited_points=sum(int(r['audited_points']) for r in rows),
                singleton_cells=sum(r['lo']==r['hi'] for r in rows),
                slopes=sorted({int(r['slope']) for r in rows}),
                affected_snapshots=len(affected),jump_transitions=len(jumps),
                positive_jumps=sum(F(j['jump'])>0 for j in jumps),
                negative_jumps=sum(F(j['jump'])<0 for j in jumps),
                max_absolute_jump=str(max((abs(F(j['jump'])) for j in jumps),default=F(0))),
                jumps=jumps)


def minimal_counterexample():
    """Three abstract requests; object costs are independent model inputs."""
    class Toy:
        initial=set()
        window=2
        budget=2
        candidates=({0:()}, {1:('D',)}, {1:('A','B')})
        lengths=(1,2,1)
        objects={'A':dict(bytes=1,tier='RAM'),'B':dict(bytes=1,tier='RAM'),
                 'D':dict(bytes=3,tier='RAM')}
        _suffix_times=({0:F(10)}, {1:F(5)}, {1:F(0)})
        _transfer_times={'A':F(1),'B':F(4),'D':F(1)}
        def size(self,objects):
            return sum(self.objects[o]['bytes'] for o in set(objects))
    state=(F(0),0,3,{2:{'A','B'}},set(),None,(F(1),'A'),(),set(),None,{})
    cells=partition(Toy(),state,(0,1,1),{},dict(ram_to_device=F(1)),F(1,1000),F(20))
    for domain,ends,events in cells:
        score=sum(ends,Affine())
        points={(domain.lo+domain.hi)/2}
        if domain.left:
            points.add(domain.lo)
        if domain.right:
            points.add(domain.hi)
        for r in points:
            expected=F(42) if r<=6 else 30+2*r if r<10 else 22+2*r
            assert score.at(r)==expected
        yield dict(lo=str(domain.lo),hi=str(domain.hi),left=int(domain.left),right=int(domain.right),
                   intercept=str(score.a),slope=str(score.b),events=str(events))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=7680)
    p.add_argument('--rounds',type=int,default=500)
    p.add_argument('--extended',action='store_true')
    p.add_argument('--witness',type=Path)
    p.add_argument('--replay',type=Path)
    p.add_argument('--summary',type=Path)
    p.add_argument('--minimal',action='store_true')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    count=0
    with args.out.open('x',newline='',encoding='utf-8') as f:
        writer=None
        rows=(minimal_counterexample() if args.minimal else
              replay_witness(args.replay) if args.replay else study(args))
        for row in rows:
            if writer is None:
                writer=csv.DictWriter(f,fieldnames=row)
                writer.writeheader()
            writer.writerow(row)
            count+=1
    print(count,'witness pairs' if args.replay else 'certified cells')
    if args.summary:
        if args.minimal or args.replay:
            raise ValueError('Summary requires parametric study output')
        with args.summary.open('x',encoding='utf-8') as f:
            json.dump(summarize(args.out),f,indent=2)


if __name__=='__main__':
    main()
