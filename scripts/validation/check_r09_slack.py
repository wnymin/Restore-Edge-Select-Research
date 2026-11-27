"""R09-CPU-01 checks for exact counterfactual timing and unchanged default execution."""
from fractions import Fraction as F

from scripts.cpu.replay import study_r08_capacity as capacity
from scripts.cpu.replay.study_r09_slack import anchor, effects, fixed, prefix_cap, stratum


def main():
    model, scale = capacity.models(capacity.read(capacity.R07_CONFIG))['4b']

    def experiment(label, cap):
        return capacity.open_exp(capacity.micro_fixture(label), model, 16, cap*capacity.MIB,scale)

    plan = [0, 576, 576]
    for label in ('same-time','stagger','shared-kv','late'):
        for cap in (64,256):
            exp = experiment(label,cap)
            anchor(exp)
            ordinary = exp.run('fixed',boundaries=plan)
            instrumented = fixed(exp,plan,[0]*3,{})
            assert {k:v for k,v in instrumented.items() if k not in ('decision_timing','decision_intervals_ms')} == ordinary
            try:
                fixed(exp,plan,[0]*3,{1:F(-1)})
            except ValueError:
                pass
            else:
                raise AssertionError('Release before decision eligibility')
    exp = experiment('same-time',256)
    b = fixed(exp,plan,[0]*3,{})
    assert b['prefetch_reserved_at_ms'][1:] == [F(1), F(1)]
    assert b['decision_timing'][1]['running_until_ms'] > 1
    e = effects(exp,b,fixed(exp,plan,[0]*3,{1:F(1,10)}))
    assert e['perf_safe'] and not e['events_safe'] and e['reservation_late'] == 2
    # Two individually hidden delays can conflict on the single decision lane.
    for idx in (1,2):
        assert effects(exp,b,fixed(exp,plan,[0]*3,{idx:F(640)}))['perf_safe']
    assert not effects(exp,b,fixed(exp,plan,[0]*3,{1:F(640),2:F(640)}))['perf_safe']
    exp = experiment('late',256)
    b = fixed(exp,plan,[0]*3,{})
    assert stratum(b['decision_timing'][2])=='idle/cold-issuable'
    e = effects(exp,b,fixed(exp,plan,[0]*3,{2:F(1,20)}))
    assert e['compute_late']==e['completion_late']==1 and not e['perf_safe']
    exp = experiment('stagger',256)
    b = fixed(exp,plan,[0]*3,{})
    assert stratum(b['decision_timing'][2])=='busy/cold-lane'
    exp = experiment('same-time',64)
    b = fixed(exp,plan,[0]*3,{})
    assert stratum(b['decision_timing'][1])=='busy/cold-capacity'
    e = effects(exp,b,fixed(exp,plan,[0]*3,{1:F(1)}))
    assert e['perf_safe'] and e['events_safe']
    assert prefix_cap([dict(delay_ms=str(i),pass_=v) for i,v in enumerate((True,True,False,True))],'pass_')==(1,2)
    print('R09 fixture checks: zero and timed anchor equivalence, GPU/IO/capacity deadlines, exact ties and shared CPU budget')


if __name__=='__main__':
    main()
