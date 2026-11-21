"""Measure closed-loop response with actual selector runtime injected."""
import argparse
from fractions import Fraction as F
import json
import platform
from random import Random
from statistics import median

from scripts.cpu.lookahead.screening.study_lookahead_trigger import Trigger
from scripts.cpu.lookahead.staging.study_r02_reference import jsonable
from scripts.validation.study_r03_capacity import ROOT, read


def measure(repeats, target=False):
    model = json.loads((ROOT/'configs/qwen3.5-4b.json').read_text(encoding='utf-8-sig'))['models'][0]
    rng = Random(930302)
    rows = []
    for case in read('r02-cpu-inputs.json')['cases']:
        if case['id'].split('-')[2] != '+0':
            continue
        exp = Trigger(case['trace'], model, F(case['capacity_bytes'], 1 << 20))
        for multiplier in ((F(1),) if target else (F(1, 2), F(2))):
            biased = {} if target else {'ssd_to_device': multiplier}
            def run(bias):
                return exp.run_trigger(biased if bias else {}, 'target' if target and bias else 'full', measure=True)

            for bias in (False, True):
                run(bias)
            first = rng.randrange(2)
            for trial in range(repeats):
                order = (False, True) if (trial + first) % 2 == 0 else (True, False)
                results = {}
                for bias in order:
                    result, _ = run(bias)
                    results[bias] = result
                base, other = (results[key] for key in (False, True))
                old, new = [[t-a for t,a in zip(r['completion_ms'],exp.arrivals,strict=True)] for r in (base,other)]
                rows.append(dict(id=case['id'], split=case['split'], multiplier=str(multiplier), trial=trial,
                    baseline_first=not order[0], baseline_actions=base['boundaries_tokens'],
                    biased_actions=other['boundaries_tokens'], baseline_response=old, biased_response=new,
                    baseline_decision_ns=sum(ns for _,ns in base['decision_profile_ns']),
                    biased_decision_ns=sum(ns for _,ns in other['decision_profile_ns']),
                    net_mean_loss_ms=(sum(new)-sum(old))/len(old), p95_loss_ms=max(new)-max(old),
                    max_individual_loss_ms=max(F(0),*(b-a for a,b in zip(old,new,strict=True)))))
    return rows


def summarize(rows):
    output = []
    for split in ('design','holdout'):
        for multiplier in sorted({r['multiplier'] for r in rows}):
            group = [r for r in rows if r['split']==split and r['multiplier']==multiplier]
            ids = sorted({r['id'] for r in group})
            cases = [median(r['net_mean_loss_ms'] for r in group if r['id']==i) for i in ids]
            output.append(dict(split=split, multiplier=multiplier, cases=len(ids), pairs=len(group),
                 case_average_net_mean_loss_ms=sum(cases)/len(cases),
                 median_decision_extra_ns=median(r['biased_decision_ns']-r['baseline_decision_ns'] for r in group),
                 maximum_individual_loss_ms=max(r['max_individual_loss_ms'] for r in group)))
    return output


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=8)
    parser.add_argument('--target', action='store_true')
    args=parser.parse_args()
    if args.repeats < 2 or args.repeats % 2:
        parser.error('Use an even number of repetitions, at least two')
    rows=measure(args.repeats, args.target)
    print(json.dumps(jsonable(dict(seed=930302, python=platform.python_version(),
                                   repeats=args.repeats, comparison='target' if args.target else 'ssd-estimate', summary=summarize(rows), rows=rows)),indent=2))
