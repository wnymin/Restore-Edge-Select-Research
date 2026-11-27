"""R09 sensitivity of fixed-action slack to six measured anchor cost records."""
import argparse
from datetime import datetime, timedelta, timezone
from fractions import Fraction as F
import json
from pathlib import Path
import platform
from statistics import median
from time import get_clock_info

from scripts.cpu.replay.study_r09_budget import assess_rule, budget_rule, capacity, workloads
from scripts.cpu.replay.study_r09_slack import anchor


def run(inputs, lock):
    manifest = capacity.read(inputs)
    b = capacity.read(lock)
    assert b['inputs_sha256']==capacity.file_hash(inputs)
    cases = [a for a in manifest['calibration'] if a['model']=='4b-native']
    for t in {a['trace'] for a in cases}:
        assert capacity.file_hash(capacity.TRACE_DIR/t)==manifest['trace_sha256'][t]
    budget_caps = sorted({F('.05'),F(b['release_budget_ms']),*map(F,b['experimental_caps_ms'])})
    rows = []
    for key,exp in workloads(dict(cases=cases,strata=manifest['strata'])):
        anchors = []
        for _ in range(6):
            sampled,plan,durations = anchor(exp)
            ns = [s['measured_ns'] for s in sampled['decision_timing']]
            row = dict(key=key,plan=plan,sample_ns=ns)
            origin = budget_rule(exp,plan,ns,0)
            assert origin['completion_ms']==sampled['completion_ms']
            anchors.append(dict(sampled_total_ns=sum(ns),measured_ns=ns,plan=plan,cap={str(cap):
                assess_rule(exp,row,cap,origin=origin)['perf_safe'] for cap in budget_caps}))
        rows.append(dict(case=key,anchors=anchors))
    status = {str(cap):dict(unanimous_pass=sum(all(a['cap'][str(cap)] for a in r['anchors']) for r in rows),
                            unanimous_fail=sum(not any(a['cap'][str(cap)] for a in r['anchors']) for r in rows),
                            flips=sum(len({a['cap'][str(cap)] for a in r['anchors']})>1 for r in rows)) for cap in budget_caps}
    info = get_clock_info('perf_counter')
    return dict(recorded=datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds'),
                inputs_sha256=capacity.file_hash(inputs),lock_sha256=capacity.file_hash(lock),
                clock=dict(implementation=info.implementation,resolution_seconds=info.resolution,
                           OS=platform.platform(),python=platform.python_version()),
                calibration_configs=len(rows),anchor_rounds=6,remaining=0,
                median_anchor_total_spread_ms=median((max(a['sampled_total_ns'] for a in r['anchors'])-
                       min(a['sampled_total_ns'] for a in r['anchors']))/1e6 for r in rows),status=status,rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('inputs','lock','out'):
        p.add_argument(k,type=Path)
    a=p.parse_args()
    d=run(a.inputs,a.lock)
    with a.out.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(d,f,indent=2)
        f.write('\n')


if __name__=='__main__':
    main()
