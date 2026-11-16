"""Aggregate paired replay metrics or decision-time routing evidence."""

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import json
from pathlib import Path
from statistics import mean
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--routes', action='store_true')
    args = parser.parse_args()
    out = csv.writer(sys.stdout, lineterminator='\n')
    cols = ('decisions', 'certified', 'screened', 'changed', 'replays', 'gap_max_ms',
            'predicted_gain_sum_ms', 'audited_screened', 'screen_misses', 'screen_missed_gain_sum_ms')
    out.writerow(('run', 'io_scale', 'policy', 'route', *cols) if args.routes else (
        'run', 'io_scale', 'policy', 'cases', 'requests', 'mean_gain_ms', 'mean_p95_gain_ms',
        'timed_gain_ms', 'timed_p95_gain_ms', 'mean_losses_gt1ms', 'p95_losses_gt10ms',
        'requests_slower_gt10ms', 'timed_requests_slower_gt10ms', 'max_request_loss_ms',
        'decision_replays', 'forecast_events', 'decision_cpu_ms_per_case'))
    for path in args.inputs:
        groups = defaultdict(list)
        with path.open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                groups[Fraction(row['io_scale']), row['policy']].append(row)
        if not groups or any(len(rows) != len(next(iter(groups.values()))) for rows in groups.values()):
            parser.error(f'Incomplete matrix: {path}')
        for (rate, policy), rows in sorted(groups.items()):
            if args.routes:
                routes = defaultdict(Counter)
                for row in rows:
                    for label, data in json.loads(row['adaptive_routes']).items():
                        tally = routes[label]
                        for k, v in data.items():
                            tally[k] = max(tally[k], v) if k == 'gap_max_ms' else tally[k] + v
                for label, tally in sorted(routes.items()):
                    out.writerow((path.stem, str(rate), policy, label, *(round(tally[k], 6) for k in cols)))
                continue
            def values(k):
                return [float(r[k]) for r in rows if r[k]]

            def avg(k):
                data = values(k)
                return round(mean(data), 6) if data else ''

            def total(k):
                data = values(k)
                return int(sum(data)) if data else ''

            out.writerow((path.stem, str(rate), policy, len(rows), total('requests'), avg('mean_gain_ms'),
                          avg('p95_gain_ms'), avg('timed_gain_ms'), avg('timed_p95_ms'),
                          sum(v < -1 for v in values('mean_gain_ms')), sum(v < -10 for v in values('p95_gain_ms')),
                          total('requests_slower_gt10ms'), total('timed_requests_slower_gt10ms'),
                          round(max(values('max_request_loss_ms')), 6), total('replays'),
                          total('forecast_events'), avg('decision_cpu_ms')))


if __name__ == '__main__':
    main()
