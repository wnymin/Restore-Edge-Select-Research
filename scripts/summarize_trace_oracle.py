"""Aggregate per-configuration offline bounds using equally weighted configurations."""

import csv
from fractions import Fraction as F
from itertools import groupby
from pathlib import Path
from statistics import mean
import sys

from study_trace_oracle import fraction


def main(path, output):
    with path.open(encoding='utf-8', newline='') as source:
        rows = list(csv.DictReader(source))
    key = lambda r: (r['run'], F(r['io_scale']), r['policy'])
    with output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator=chr(10))
        writer.writerow(('run', 'io_scale', 'policy', 'cases', 'exact_cases', 'no_headroom_cases',
                         'baseline_ms', 'strategy_ms', 'optimum_lower_ms', 'optimum_upper_ms',
                         'headroom_lower_ms', 'headroom_upper_ms', 'captured_lower_pct',
                         'captured_upper_pct', 'ratio_status'))
        for k, group in groupby(sorted(rows, key=key), key=key):
            a = list(group)
            base, strategy, lo, hi = (mean(F(r[c]) for r in a) for c in
                                      ('baseline_ms', 'strategy_ms', 'optimum_lower_ms', 'optimum_upper_ms'))
            ratio_lo, ratio_hi, status = fraction(base, strategy, lo, hi)
            writer.writerow((*k, len(a), sum(int(r['exact']) for r in a),
                             sum(r['ratio_status'].startswith('no-headroom') for r in a),
                             *(round(float(v), 6) for v in (base, strategy, lo, hi, base - hi, base - lo)),
                             *(round(v, 6) if isinstance(v, float) else v for v in (ratio_lo, ratio_hi)), status))


if __name__ == '__main__':
    main(*map(Path, sys.argv[1:]))
