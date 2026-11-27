"""Check R08-A01-type downstream-action/earlier-request coupling on the new windows."""
import argparse
from fractions import Fraction as F
import gzip
import json
from pathlib import Path

from scripts.validation.run_r08_sealed_holdout import BASE, CANDIDATE, LOCAL, arrivals, capacity, native, workloads


def study(seal, raw):
    m = capacity.read(seal)
    receipt = capacity.read(raw.with_name(raw.name + '.manifest.json'))
    assert receipt['sealed_sha256'] == capacity.file_hash(seal)
    assert receipt['output_sha256'] == capacity.file_hash(raw)
    cases, screened = [], 0
    with gzip.open(raw, 'rt', encoding='utf-8') as f:
        selected = []
        for row in map(json.loads, f):
            a, b = (row['zero'][p] for p in (BASE, LOCAL))
            if not row['repeat_tail'][BASE]:
                continue
            screened += 1
            target = next((i for i,(x,y) in enumerate(zip(a['plan'],b['plan'],strict=True)) if x!=y),None)
            hit = [i for i in row['repeat_tail'][BASE] if target is not None and i<target and a['extra_vs_local_ms'][i]>10]
            if hit:
                selected.append((row,target,hit))
    profiles = capacity.models(capacity.read(capacity.R07_CONFIG)) | native.model_cases()
    for (row,target,hit),(case,exp) in zip(selected,workloads(dict(m,cases=[r['key'] for r,_,_ in selected])),strict=True):
        assert row['key'] == case
        base, local = [arrivals.run(exp,p) for p in (BASE,LOCAL)]
        plan, other = (r['boundaries_tokens'] for r in (base,local))
        assert list(plan) == row['zero'][BASE]['plan'] and list(other) == row['zero'][LOCAL]['plan']
        forced = arrivals.run(exp,BASE,overrides={target:other[target]})
        reverse = arrivals.run(exp,LOCAL,overrides={target:plan[target]})
        # Hold both complete action sequences fixed for the overlap ablation.
        no_overlap = capacity.Experiment(dict(exp.trace,overlap_slowdown={}),profiles[case['model']][0],F(exp.budget,capacity.MIB))
        off = [no_overlap.run('fixed',boundaries=z['boundaries_tokens']) for z in (base,forced)]
        b,l,f,r,x,y = [arrivals.response(exp,z) for z in (base,local,forced,reverse,*off)]
        hits = [dict(index=i,request_id=row['request_ids'][i],frozen_extra_ms=float(b[i]-l[i]),
                     forced_extra_ms=float(f[i]-l[i]),reverse_extra_ms=float(r[i]-l[i]),
                     no_overlap_action_delta_ms=float(x[i]-y[i]),
                     matched=f[i]==l[i] and r[i]==b[i] and x[i]==y[i],
                     candidate_still_repeated=i in row['repeat_tail'][CANDIDATE]) for i in hit]
        cases.append(dict(case=case,first_split=target,frozen_action=plan[target],forced_action=other[target],
                          mean_cost_of_forcing_ms=float(forced['response_mean_ms']-base['response_mean_ms']),hits=hits))
    return dict(sealed_sha256=receipt['sealed_sha256'],output_sha256=receipt['output_sha256'],
                frozen_repeat_configurations=screened,cases=cases)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ('seal','raw','out'):
        p.add_argument(field,type=Path)
    a = p.parse_args()
    data = study(a.seal,a.raw)
    with a.out.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(data,f,indent=2)
        f.write('\n')


if __name__=='__main__':
    main()
