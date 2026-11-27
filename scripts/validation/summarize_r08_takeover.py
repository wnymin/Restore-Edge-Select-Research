"""Verify and summarize the prespecified same-team R08 holdout."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import gzip
import json
from pathlib import Path

from scripts.validation.check_r08_refinement import BASE, CANDIDATE, LOCAL, case_metrics
from scripts.validation.run_r08_sealed_holdout import capacity


def load_cases(seal, raw):
    m = capacity.read(seal)
    receipt = capacity.read(raw.with_name(raw.name + '.manifest.json'))
    assert receipt['sealed_sha256'] == capacity.file_hash(seal)
    assert receipt['output_sha256'] == capacity.file_hash(raw)
    for field in ('frozen_sha256', 'driver_sha256'):
        assert receipt[field] == m[field]

    def key(case):
        return (case['trace'], case['window'], case['stratum'], case['model'],
                case['rate'], case['capacity_bytes'])

    expected = {key(case): case for case in m['cases']}
    with gzip.open(raw, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            assert row['key'] == expected.pop(key(row['key']))
            summary = case_metrics(row)
            lo, hi = (F(summary['offline'][k]) for k in ('lower', 'upper'))
            local, base = (F(row['zero'][p]['mean_exact']) for p in (LOCAL, BASE))
            # The two frozen-gap criteria share the same unknown offline optimum.
            hi = min(hi, local, base)
            assert lo <= hi <= base
            summary['certified_frozen_gap'] = (local > hi and base-hi > 1
                                                   and (base-hi)*20 > local-hi)
            summary['possible_frozen_gap'] = local > lo and base-lo > 1 and (base-lo)*20 > local-lo
            summary['request_ids'] = row['request_ids']
            summary['zero_metrics'] = {p:{k:row['zero'][p][k] for k in ('mean_exact','p95','max_local_extra','late')}
                                       for p in (LOCAL, BASE, CANDIDATE)}
            yield summary
    assert not expected


def stratum(cases, t):
    counts = Counter(exact=sum(r['offline']['exact'] for r in cases))
    sources = defaultdict(lambda: defaultdict(list))
    hard = []
    for r in cases:
        k = r['key']
        f = r['flags']
        counts.update(name for name, flag in f.items() if flag)
        for metric in ('certified_frozen_gap','possible_frozen_gap'):
            counts[metric] += r[metric]
        counts['changed'] += r['routes'].get('changed', 0) > 0
        failures = [v for v in ('new_repeated_tail','zero_opt_fail') if f[v]]
        if f['good']:
            failures += [v for v in ('mean_fail','p95_fail','retention_fail') if f[v]]
            data = sources[k['trace'], k['window']]
            for field in ('mean','p95'):
                data[field].append(r['timed_median'][field])
            if r['net_retention'] is not None:
                data['retention'].append(r['net_retention'])
        if failures:
            hard.append(dict(case=k, failures=failures))
    limits = {}
    source_metrics = defaultdict(dict)
    for metric, target in (('mean',t['mean_ms']),('p95',t['p95_ms']),('retention',t['keep_net'])):
        extreme = min if metric == 'retention' else max
        for source, data in sources.items():
            if data.get(metric):
                source_metrics[source][metric] = extreme(data[metric])
        values = [v[metric] for v in source_metrics.values() if metric in v]
        value = extreme(values) if values else None
        within = value is not None and (value >= target if metric == 'retention' else value <= target)
        n = len(values)
        limits[metric] = dict(source_count=n, worst_observed=value, limit=target,
                              one_sided_median_bound=value if n>=5 else None,
                              nominal_coverage=1-2**(-n) if n>=5 else None,
                              outcome='pass' if n>=5 and within else 'insufficient')
    return dict(case_count=len(cases), source_count=len({(r['key']['trace'],r['key']['window']) for r in cases}),
                counts=dict(counts), bounds=limits, hard_failures=hard,
                source_metrics=[dict(trace=t,window=w,**v) for (t,w),v in sorted(source_metrics.items())],
                gate2='fail' if hard else ('pass' if all(v['outcome']=='pass' for v in limits.values()) else 'insufficient'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ('seal','raw','out'):
        p.add_argument(field, type=Path)
    a = p.parse_args()
    m = capacity.read(a.seal)
    rows = list(load_cases(a.seal, a.raw))
    receipt = capacity.read(a.raw.with_name(a.raw.name + '.manifest.json'))
    assert len(rows) == receipt['n'] == len(m['cases'])
    stats = {s:stratum([r for r in rows if r['key']['stratum']==s],m['thresholds']) for s in m['strata']}
    evidence = dict(protocol='same-team takeover; not independent', receipt=receipt, strata=stats, cases=rows)
    with a.out.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(evidence, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
