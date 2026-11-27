"""Localize long same-policy responses using existing native/controller logs."""

import argparse
from collections import defaultdict
import csv
from fractions import Fraction
from io import TextIOWrapper
import json
from math import ceil
from pathlib import Path
from statistics import median
from zipfile import ZipFile

from scripts.validation.check_r10_native_catalog import digest


def range_ms(values):
    return dict(n=len(values), min_ms=min(values), median_ms=median(values), max_ms=max(values))


def arm(archive, run_name, trace):
    receipt = json.loads(archive.read(run_name))
    with archive.open(run_name.removesuffix('.run.json') + '.native.csv') as file:
        rows = [dict(event=x.pop('event'), **{k: int(v) for k, v in x.items()})
                for x in csv.DictReader(TextIOWrapper(file, encoding='ascii'))]
    stages, copy = defaultdict(dict), defaultdict(list)
    for x in rows:
        if x['event'] in ('native-target', 'suffix-submitted', 'compute-done'):
            assert x['event'] not in stages[x['request']]
            stages[x['request']][x['event']] = x['native_us']
        if x['event'] in ('native-scatter', 'preserve-recurrent', 'preserve-old-active'):
            assert x['interval_us'] > 0
            copy[f"{x['event']}:{x['bytes']}"].append(x['interval_us'] / 1000)
    keys = ['client_arrivals_ms', 'observed_arrivals_ms', 'completions_ms']
    stamps = [list(map(Fraction, receipt[k])) for k in keys]
    assert len(stages) == len(trace) == len(receipt['starts']) and all(len(x) == len(trace) for x in stamps)
    detailed, offsets, lengths, delays = [], [], [], []
    for i, (a, o, c) in enumerate(zip(*stamps, strict=True)):
        # A constant offset cancels when taking the span of differences.
        # The absolute difference between epochs is not a transport delay.
        assert a <= o <= c
        x = stages[i]
        done = x['compute-done']
        offsets.append(float(c - Fraction(done, 1000)))
        delays.append(float(o - a))
        if 'suffix-submitted' not in x:
            continue
        start = x['suffix-submitted']
        assert x['native-target'] <= start <= done
        elapsed = (done - start) / 1000
        lengths.append(elapsed)
        if elapsed >= 1000:  # reporting threshold only; no requests are removed
            selected = receipt['starts'][i]['compute']['boundary']
            detailed.append(dict(request=i, boundary=selected, input_tokens=sum(r['count'] for r in trace[i]['input_token_runs']),
                                 native_first_marker_to_finish_ms=elapsed, response_ms=float(c - a),
                                 arrival_to_controller_ms=float(o - a)))
    return dict(run=run_name, requests=len(trace), native_post_first_marker=range_ms(lengths),
                arrival_to_controller_p95_ms=sorted(delays)[ceil(.95 * len(delays)) - 1],
                native_controller_finish_difference_span_ms=max(offsets) - min(offsets),
                reported_post_marker_at_least_1000ms=detailed, copies={k: range_ms(v) for k, v in copy.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'aa-summary', 'schedule', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    args = ap.parse_args()
    summary = json.loads(args.aa_summary.read_bytes())
    trace = json.loads(args.schedule.read_bytes())['trace']['requests']
    archive_hash = digest(args.archive)
    if archive_hash != summary['archive_sha256']:
        raise ValueError('timing audit requires the archive audited by the A/A summary')
    with ZipFile(args.archive) as archive:
        names = sorted(x for x in archive.namelist() if x.endswith('.run.json'))
        assert len(names) == 2
        result = dict(archive_sha256=archive_hash, schedule_sha256=digest(args.schedule),
                      scope='latency localization, not device busy time or a strategy effect',
                      arms=[arm(archive, name, trace) for name in names])
    with args.out.open('x', encoding='utf8', newline='\n') as out:
        print(json.dumps(result, indent=2, sort_keys=True), file=out)


if __name__ == '__main__':
    main()
