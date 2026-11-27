"""Controlled R09 deadline cases; no dependence on live wall-clock jitter."""
from fractions import Fraction as F
from unittest.mock import patch

from scripts.cpu.replay import study_r09_quality as q


def main():
    args=(1,0,2,512,False,True,set(),set(),lambda _:0,lambda _:(0,0))
    path='scripts.cpu.replay.study_r09_quality.perf_counter_ns'

    def run(stamps,*,running=True, guard=5):
        obj=q.Budgeted(None,q.Refinement(),F(1,1000),guard)  # 1000 ns
        obj.refine=lambda *args: (args[-1]((512,)),1024)[-1]
        with patch(path,side_effect=iter(stamps)):
            return obj(*(*args[:5],running,*args[6:])),obj.stats

    assert run([0,100,200,300])[0]==1024
    chosen,stat=run([0,100,1001,1200])
    assert chosen==512 and stat['budget_abort']==stat['budget_overrun']==1
    chosen,stat=run([0,998,1003],guard=5)
    assert chosen==512 and stat['budget_abort']==stat['budget_overrun']==1
    chosen,stat=run([0],running=False)
    assert chosen==512 and not stat
    print('R09 wall-clock budget: completed action, fallback before/after atomic step, no idle search')


if __name__=='__main__':
    main()
