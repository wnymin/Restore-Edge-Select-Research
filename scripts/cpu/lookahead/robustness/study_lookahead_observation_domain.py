"""Exact observation-conditioned residual sets, with a common uncertain clock.

These are conditional mathematical bounds, not calibrated hardware intervals.
Both active flags are exact post-event observations; zero residual is excluded.
"""

import argparse
from fractions import Fraction as F
import json
from pathlib import Path
from random import Random

from scripts.validation.check_joint_domain_evidence import inside
from scripts.cpu.lookahead.robustness.study_lookahead_joint_domain import Linear, Region
from scripts.cpu.lookahead.staging.study_lookahead_two_request import jsonable


def total_interval(predicted, relative_error):
    """Assume abs(predicted - actual) <= eta * actual, not eta * predicted."""
    assert predicted > 0 and 0 <= relative_error < 1
    return predicted/(1+relative_error),predicted/(1-relative_error)


def residual_domain(compute_end,transfer_end,clock):
    """Absolute end-time bounds and one common clock interval; both active."""
    a,b=compute_end
    c,d=transfer_end
    lo,hi=clock
    assert a<=b and c<=d and lo<=hi
    rlo,rhi=max(F(0),a-hi),b-lo
    hlo,hhi=max(F(0),c-hi),d-lo
    if rhi<=0 or hhi<=0:
        return None
    # Degenerate rectangle/segment/singleton supported by Region clipping.
    vertices=tuple(dict.fromkeys(((rlo,hlo),(rhi,hlo),(rhi,hhi),(rlo,hhi))))
    region=Region(vertices)
    constraints=(Linear(0,-1,0),Linear(0,0,-1),
                 Linear(a-d,-1,1),Linear(c-b,1,-1))
    for i,line in enumerate(constraints):
        region=region.clip(line,strict=i<2)
        if region is None:
            return None
    return region


def feasible(point,compute_end,transfer_end,clock):
    r,h=point
    if r<=0 or h<=0:
        return False
    # Independent witness for existence of one common observation time.
    lower=max(clock[0],compute_end[0]-r,transfer_end[0]-h)
    upper=min(clock[1],compute_end[1]-r,transfer_end[1]-h)
    return lower<=upper


def native_segment():
    from scripts.cpu.lookahead.robustness.study_lookahead_joint_domain import Capture, ROOT, analyze, read
    source=read(ROOT/'docs/evidence/dynamic-lookahead-transfer-order-7820.json')['cases'][0]
    trace=source['trace']
    for req in trace['requests'][1:]:
        req['arrival_ms']=0
    model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    exp=Capture(trace,model,80)
    exp.run_trigger({},'full')
    first=next(o for o in exp.candidates[2][1152] if o not in exp.initial)
    duration=exp._transfer_times[first]
    now=duration/2
    end=F(225,8)
    clock=(duration/4,duration*F(3,4))
    domain=residual_domain((end,end),(duration,duration),clock)
    assert domain is not None and domain.dimension()==1
    state=(now,0,3,{2:set(exp.candidates[2][1152])-exp.initial},set(exp.initial),
           end,(duration,first),(),set(),None,{})
    record=(1,0,state,(0,0,1152),{},dict(ram_to_device=F(1),ssd_to_device=F(1)))
    rows=analyze(exp,record,rectangle=domain,parameter='active-pair')
    assert all(r['faces']==0 for r in rows)
    return jsonable(dict(clock=clock,exact_residual_segment=domain.vertices,rows=rows))


def run():
    rng=Random(7860)
    cases=[]
    grid_checks=coverage_checks=0
    for index in range(200):
        alo=F(rng.randrange(0,16)); ahi=alo+rng.randrange(0,9)
        blo=F(rng.randrange(0,16)); bhi=blo+rng.randrange(0,9)
        tlo=F(rng.randrange(0,12)); thi=tlo+rng.randrange(0,6)
        bounds=((alo,ahi),(blo,bhi),(tlo,thi))
        region=residual_domain(*bounds)
        for i in range(17):
            for j in range(17):
                point=F(i),F(j)
                actual=region is not None and inside(region,point)
                assert actual==feasible(point,*bounds),(bounds,point,actual)
                grid_checks+=1
        if region is not None:
            for point in (*region.vertices,region.center()):
                assert inside(region,point)==feasible(point,*bounds)
        # Construct actual histories, then predict with bounded total error.
        wa=F(rng.randrange(1,41)); wb=F(rng.randrange(1,41))
        sa=F(rng.randrange(0,4)); sb=F(rng.randrange(0,4))
        end_a,end_b=sa+wa,sb+wb
        now=min(end_a,end_b)/2
        eta=F(1,4)
        pa=wa*(1+rng.choice((-eta,F(0),eta)))
        pb=wb*(1+rng.choice((-eta,F(0),eta)))
        la,ua=total_interval(pa,eta); lb,ub=total_interval(pb,eta)
        clock=(max(sa,sb,now-F(1,4)),now+F(1,4))
        # If starts happen after this chosen now, skip this history.
        if now>=max(sa,sb):
            domain=residual_domain((sa+la,sa+ua),(sb+lb,sb+ub),clock)
            true=(end_a-now,end_b-now)
            assert domain is not None and inside(domain,true)
            coverage_checks+=1
        cases.append(jsonable(dict(case=index,bounds=bounds,empty=region is None,
                                   closure=None if region is None else region.vertices)))
    # Exact total costs plus uncertain common time: only a diagonal segment.
    segment=residual_domain((F(10),F(10)),(F(12),F(12)),(F(5),F(7)))
    assert segment.dimension()==1
    assert set(segment.vertices)=={(F(3),F(5)),(F(5),F(7))}
    assert not feasible((F(3),F(7)),(F(10),F(10)),(F(12),F(12)),(F(5),F(7)))
    # Still-active observation contradicts an end interval entirely in the past.
    assert residual_domain((F(1),F(2)),(F(8),F(9)),(F(3),F(4))) is None
    # A zero closure endpoint is a limit, not an active zero state.
    active=residual_domain((F(2),F(4)),(F(3),F(5)),(F(3),F(3)))
    assert active is not None and not inside(active,(F(0),F(1)))
    lo,hi=total_interval(F(10),F(1,4))
    assert (lo,hi)==(F(8),F(40,3))
    return jsonable(dict(seed=7860,cases=cases,grid_checks=grid_checks,native_segment=native_segment(),
                         conditional_coverage_checks=coverage_checks,
                         common_clock_witness=dict(compute_end=10,transfer_end=12,clock=[5,7],
                                                   exact_segment=segment.vertices,
                                                   impossible_box_corner=[3,7],
                                                   box_sup_h_minus_r=4,joint_sup_h_minus_r=2),
                         relative_error_example=dict(predicted=10,eta='1/4',actual_total=[lo,hi])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--check',type=Path)
    args=parser.parse_args()
    result=run()
    if args.check:
        with args.check.open(encoding='utf-8') as f:
            assert result==json.load(f)
    if args.out:
        with args.out.open('x',encoding='utf-8') as f:
            json.dump(result,f,ensure_ascii=False,indent=2)
    print({k:v for k,v in result.items() if k not in ('cases','common_clock_witness','relative_error_example')})
