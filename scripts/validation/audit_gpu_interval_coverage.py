"""Retrospective frozen empirical envelope audit, not a safety calibration."""

import argparse
from collections import defaultdict
import csv
from decimal import Decimal as D
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def rows(name):
    path=ROOT/'docs/evidence'/name
    result=[]
    with path.open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if int(row['rep'])>=0 and row['mode']=='distinct' and int(row['blocked'])==1:
                assert row['arm']=='deep' and row['quota_mib']=='80'
                value=D(row['peer_d_ready_ms'])-D(row['release_ms'])
                assert value>0
                result.append((row,value))
    assert result
    return result


def run():
    cal='dynamic-gpu-joint-staging-7449.csv'
    calibration=rows(cal)
    # Freeze both endpoints using calibration only, before loading later batches.
    low=min(v for _,v in calibration)
    high=max(v for _,v in calibration)
    groups=[]
    for name in (cal,'dynamic-gpu-joint-staging-7450.csv','dynamic-gpu-joint-staging-7451.csv',
                 'dynamic-gpu-lookahead-critical-7620.csv','dynamic-gpu-lookahead-critical-7621.csv'):
        data=rows(name)
        by_work=defaultdict(list)
        for row,value in data:
            by_work[int(row['shallow_work_ms'])].append(value)
        groups.append(dict(source=name,sha256=hashlib.sha256((ROOT/'docs/evidence'/name).read_bytes()).hexdigest(),
                           samples=len(data),minimum=str(min(v for _,v in data)),maximum=str(max(v for _,v in data)),
                           below=sum(v<low for _,v in data),above=sum(v>high for _,v in data),
                           works={str(k):dict(n=len(v),below=sum(x<low for x in v),above=sum(x>high for x in v))
                                  for k,v in sorted(by_work.items())}))
    return dict(quantity='peer_d_ready_ms - release_ms; effective release-to-ready, not isolated transfer',
                calibration_source=cal,frozen_minimum=str(low),frozen_maximum=str(high),groups=groups,
                limitations=['No per-decision active residual snapshot or completion-notification delay.',
                             'Observed extrema are not deterministic future service bounds.',
                             'Later batches differ in timing/work settings; no iid assumption.',
                             'Two arms of a paired request are not simultaneous counterfactual observations.'])


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
    print(json.dumps(result,ensure_ascii=False))
