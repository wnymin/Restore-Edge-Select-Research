"""Report a per-configuration offline interval and the fraction of gain captured."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from fractions import Fraction as F
from itertools import islice
import json
from math import prod
from pathlib import Path

from study_bailian import Experiment, ROOT, build, read
from study_load_routing import run_case
from restore_edge_select.offline_bounds import search

KEYS = ('cohort', 'trace', 'window', 'placement', 'spacing', 'budget', 'io_scale', 'path', 'max_prompt')
POLICIES = ('next4', 'entry', 'mixed-reuse-estimate', 'adaptive', 'adaptive-screened', 'adaptive-mean')


class Discard:
    def writerow(self, row):
        pass


def fraction(base, result, lo, hi):
    gain = base - result
    if lo == base:
        return '', '', 'no-headroom' if result == base else 'no-headroom-regression'
    if gain == 0:
        return 0, 0, 'possibly-no-headroom' if hi == base else 'ratio'
    if hi == base:
        return '-inf', float(100 * gain / (base - lo)), 'possibly-no-headroom'
    ends = sorted(100 * gain / (base - x) for x in (lo, hi))
    return *map(float, ends), 'ratio'


def window(job):
    source, key, records, limit = job
    metadata = read(source.with_suffix('.metadata.json'))
    params, inputs = metadata['parameters'], metadata['inputs']
    _, trace, number = key
    number = int(number)
    fixture = inputs['fixture']['content']
    model = inputs['model']['content']['models'][0]
    grid = inputs['grid']['content']
    io = inputs['io_profile']['content']['cost']
    overlap = inputs['overlap_profile']['content']
    estimate_factor = F(params['ssd_estimate_factor'])
    estimate = {'ssd_to_device': estimate_factor} if estimate_factor != 1 else None
    with (ROOT / 'data/raw/bailian' / trace).open(encoding='utf-8') as stream:
        rows = [json.loads(line, parse_float=str) for line in islice(stream, 352 * number, 352 * (number + 1))]
    data, output = {}, []
    for key, archive in records.items():
        k = dict(zip(KEYS, key, strict=True))
        placement, spacing = k['placement'], F(k['spacing'])
        if (placement, spacing) not in data:
            data[placement, spacing] = build(rows, 256, fixture, model, spacing, placement, int(k['max_prompt']),
                                              all_checkpoints=params.get('all_checkpoints', False))
        cost = dict(fixture['cost'], recompute=grid)
        for path, cfg in io.items():
            scaled = path == 'ssd_to_device' or (path == 'ram_to_device' and k['path'] == 'all')
            factor = F(k['io_scale']) if scaled else 1
            cost[path] = dict(startup_ms=str(F(cfg['startup_ms']) * factor),
                              bytes_per_ms=str(F(cfg['bytes_per_ms']) / factor))
        exp = Experiment(dict(data[placement, spacing], cost=cost, overlap_slowdown=overlap), model, int(k['budget']))
        policies = tuple(p for p in POLICIES if p in archive)
        if 'entry' in policies and 'next4' not in policies:
            policies = ('next4', *policies)
        captured = {}
        run_case(exp, k, Discard(), None, True, policies, estimate=estimate, capture=captured)
        base = captured['local-cost']['response_mean_ms']
        for p in policies:
            if p in archive:
                assert abs(float(base) - float(archive[p]['baseline_mean_ms'])) < 1e-7, (k, p, float(base), archive[p]['baseline_mean_ms'])
                assert abs(float(base - captured[p]['response_mean_ms']) - float(archive[p]['mean_gain_ms'])) < 1e-7, (k, p)
        n = len(exp.requests)
        combinations = prod(map(len, exp.eligible))
        lo, hi, plan, nodes = search(exp, list(captured.values()), limit)
        for p, run in captured.items():
            if p not in archive and p != 'local-cost':
                continue
            val = run['response_mean_ms']
            a, b, status = fraction(base, val, lo / n, hi / n)
            output.append(dict(run=source.stem, **k, policy=p, requests=n, combinations=combinations, search_nodes=nodes,
                               baseline_ms=float(base), strategy_ms=float(val), optimum_lower_ms=float(lo / n),
                               optimum_upper_ms=float(hi / n), exact=int(lo == hi),
                               captured_lower_pct=a, captured_upper_pct=b, ratio_status=status,
                               feasible_boundaries=json.dumps(plan, separators=(',', ':'))))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--nodes', type=int, default=100)
    args = parser.parse_args()
    if args.nodes < 0 or args.workers < 1:
        parser.error('Use a nonnegative node budget and at least one worker')
    jobs = []
    for path in args.sources:
        grouped = {}
        with path.open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                key = tuple(row[k] for k in KEYS)
                grouped.setdefault(key[:3], {}).setdefault(key, {})[row['policy']] = row
        jobs.extend((path, key, rows, args.nodes) for key, rows in grouped.items())
    def write(results):
        writer = None
        for result in results:
            if writer is None:
                writer = csv.DictWriter(out, tuple(result[0]), lineterminator='\n')
                writer.writeheader()
            writer.writerows(result)
            out.flush()

    with args.out.open('w', newline='', encoding='utf-8') as out:
        if args.workers == 1:
            write(map(window, jobs))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                write(pool.map(window, jobs))


if __name__ == '__main__':
    main()
