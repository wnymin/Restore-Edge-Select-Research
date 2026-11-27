"""Audit equal-policy full blocks and report noise without treating reruns as new traces."""

import argparse
from collections import Counter
from fractions import Fraction
import json
from math import ceil
from pathlib import Path
from statistics import mean, median
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from scripts.gpu.analysis.check_r10_catalog_fifo import check
from scripts.validation.check_r10_native_catalog import digest


def endpoints(prefix):
    return [prefix.with_suffix(x) for x in ('.run.json', '.native.csv', '.stderr')]


def resource_rows(stderr):
    values = {}
    names = ('vulkan_free_bytes', 'vulkan_reported_budget_bytes', 'working_set_bytes', 'peak_working_set_bytes', 'private_commit_bytes')
    for line in stderr.read_text(encoding='utf8', errors='replace').splitlines():
        if not line.startswith('r10_resources,'):
            continue
        _, phase, *raw = line.split(',')
        assert phase not in values
        values[phase] = dict(zip(names, map(int, raw), strict=True))
    return values


def p95(values):
    return sorted(values)[ceil(.95 * len(values)) - 1]


def one(prefix, schedule):
    source, native, stderr = endpoints(prefix)
    data = json.loads(source.read_bytes())
    assert not data['qualification'] and data['warmup'] and data['native_events'] and data['cost_profile_sha256']
    assert data['publish_rest_on'] == 'open-loop'
    score = check(source, native, schedule)
    n = score['requests']
    plan = list(map(Fraction, data['planned_arrival_offsets_ms']))
    actual = list(map(Fraction, data['client_arrivals_ms']))
    observed = list(map(Fraction, data['observed_arrivals_ms']))
    ended = list(map(Fraction, data['completions_ms']))
    assert len(plan) == len(actual) == len(observed) == len(ended) == n
    assert all(actual[i] <= observed[i] <= ended[i] for i in range(n))
    responses = [z - a for a, z in zip(actual, ended, strict=True)]
    arrival_slip = [a - p for a, p in zip(actual, plan, strict=True)]
    poll_delay = [o - a for a, o in zip(actual, observed, strict=True)]
    choices = [r['action'] for r in sorted(data['decisions'], key=lambda r: r['snapshot']['request'])]
    values = dict(response_mean_ms=float(mean(responses)), response_median_ms=float(median(responses)),
                  response_p95_ms=float(p95(responses)), max_arrival_slip_ms=float(max(arrival_slip)),
                  min_arrival_slip_ms=float(min(arrival_slip)),
                  p95_observation_delay_ms=float(p95(poll_delay)), choices=dict(Counter(choices)),
                  p95_chooser_ms=float(p95([r['choose_ns'] for r in data['decisions']]) / 1e6),
                  charged_peak_bytes=data['peak_prefetch_bytes'], resources=resource_rows(stderr),
                  actual_scatter_observations=dict(Counter({k: v for k, v in score['device_event_counts'].items()
                                                               if k in ('native-scatter', 'preserve-recurrent', 'preserve-old-active')})))
    return data, values, choices


def pair(a, b, schedule):
    first, stat_a, choose_a = one(a, schedule)
    second, stat_b, choose_b = one(b, schedule)
    invariants = ('block', 'capacity_bytes', 'worker_sha256', 'observer_sha256', 'seal_sha256', 'policy', 'planned_arrival_offsets_ms',
                  'cost_profile_sha256', 'native_events', 'warmup', 'warmup_grid', 'resources', 'chunk_events', 'pipeline_events')
    assert {k: first.get(k) for k in invariants} == {k: second.get(k) for k in invariants}
    assert first['policy'] == 'local-cost'
    return dict(block=first['block'], capacity_bytes=first['capacity_bytes'], policy=first['policy'],
                scope='same-policy full-block A/A under provisional costs; not an A/B effect estimate',
                a=stat_a, b=stat_b, choice_mismatches=sum(x != y for x, y in zip(choose_a, choose_b, strict=True)),
                abs_mean_difference_ms=abs(stat_a['response_mean_ms'] - stat_b['response_mean_ms']),
                abs_p95_difference_ms=abs(stat_a['response_p95_ms'] - stat_b['response_p95_ms']),
                cost_profile_sha256=first['cost_profile_sha256'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ('first', 'second', 'schedule', 'archive', 'out'):
        ap.add_argument('--' + k, type=Path, required=True)
    a = ap.parse_args()
    if a.archive.exists() or a.out.exists():
        raise FileExistsError('select new evidence filenames')
    result = pair(a.first, a.second, a.schedule)
    with ZipFile(a.archive, 'x') as z:
        for prefix in (a.first, a.second):
            for path in endpoints(prefix):
                z.writestr(ZipInfo(path.name), path.read_bytes(), compress_type=ZIP_DEFLATED)
    result['archive_sha256'] = digest(a.archive)
    with a.out.open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2), file=f)
    print(f"Completed A/A for {result['block']}: absolute mean difference {result['abs_mean_difference_ms']:.2f} ms.")


if __name__ == '__main__':
    main()
