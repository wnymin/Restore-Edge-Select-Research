"""Report open-loop queue and drain observations for genuine baseline runs."""

import argparse
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
from statistics import mean
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from scripts.gpu.analysis.summarize_r10_pilot_aa import endpoints, one
from scripts.validation.check_r10_native_catalog import digest


def measured_window(arrive, finish):
    assert list(arrive) == sorted(arrive) and list(finish) == sorted(finish)
    assert len(arrive) > 1 and len(arrive) == len(finish)
    assert all(a <= z for a, z in zip(arrive, finish, strict=True))
    events = defaultdict(int)
    for a, z in zip(arrive, finish, strict=True):
        events[a] += 1
        events[z] -= 1
    active = peak = at_last = 0
    area, busy_time = Fraction(), Fraction()
    previous = arrive[0]
    for time, change in sorted(events.items()):
        area += active * (time - previous)
        if active:
            busy_time += time - previous
        active += change
        assert active >= 0
        peak = max(peak, active)
        if time == arrive[-1]:
            at_last = active
        previous = time
    assert active == 0 and area == sum(z - a for a, z in zip(arrive, finish, strict=True))
    span, elapsed = arrive[-1] - arrive[0], finish[-1] - arrive[0]
    return dict(submission_span_ms=float(span), drain_elapsed_ms=float(elapsed),
                drain_after_last_arrival_ms=float(finish[-1] - arrive[-1]),
                average_outstanding=float(area / elapsed), max_outstanding=peak, outstanding_at_last_arrival=at_last,
                fraction_time_any_request_outstanding=float(busy_time / elapsed),
                offered_arrivals_per_s=float((len(arrive) - 1) * 1000 / span),
                completions_per_s_in_entire_window=float(len(arrive) * 1000 / elapsed))


def arm(prefix, schedule):
    receipt, stats, _ = one(prefix, schedule)
    assert receipt['policy'] == 'local-cost' and receipt.get('warmup_grid')
    arrive, observe, finish = [tuple(map(Fraction, receipt[k])) for k in (
        'client_arrivals_ms', 'observed_arrivals_ms', 'completions_ms')]
    return dict(block=receipt['block'], run=prefix.name, capacity_bytes=receipt['capacity_bytes'],
                source_schedule_sha256=digest(schedule), response_mean_ms=stats['response_mean_ms'],
                response_p95_ms=stats['response_p95_ms'], p95_observation_delay_ms=stats['p95_observation_delay_ms'],
                mean_observation_delay_ms=float(mean(o - a for a, o in zip(arrive, observe, strict=True))),
                window=measured_window(arrive, finish), receipt_sha256=digest(prefix.with_suffix('.run.json')))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', action='append', type=Path, nargs=2, metavar=('PREFIX', 'SCHEDULE'), required=True)
    ap.add_argument('--archive', type=Path, help='save each arm receipt, native event CSV, and stderr')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists() or (args.archive and args.archive.exists()):
        raise FileExistsError('select new load audit and archive paths')
    rows = [arm(*paths) for paths in args.arm]
    evidence = dict(scope='one finite open-loop block per row, not a steady-state throughput estimate', arms=rows)
    if args.archive:
        files = [p for prefix, _ in args.arm for p in endpoints(prefix)]
        if len({p.name for p in files}) != len(files):
            raise ValueError('all arm prefixes must have different basenames when archiving')
        with ZipFile(args.archive, 'x') as z:
            for p in files:
                z.writestr(ZipInfo(p.name), p.read_bytes(), compress_type=ZIP_DEFLATED)
        evidence['archive_sha256'] = digest(args.archive)
    with args.out.open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(evidence, indent=2, sort_keys=True), file=f)


if __name__ == '__main__':
    main()
