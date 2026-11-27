"""Audit matched native overlap and test a prespecified additive makespan correction."""

import argparse
from collections import defaultdict
import csv
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
from statistics import median, quantiles
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from scripts.validation.check_r10_native_catalog import digest


ARM = {'compute-a', 'compute-b', 'transfer-only', 'joint'}
SHAPES = {(p, n, t) for p, n in ((512, 64), (512, 512), (768, 128), (768, 384)) for t in ('RAM', 'SSD')}


def score(raw, stderr, repeats):
    samples = defaultdict(dict)
    with raw.open(encoding='ascii', newline='') as f:
        for data in csv.DictReader(f):
            tag = data.pop('arm')
            tier = data.pop('tier')
            fit = {'fit': True, 'validation': False}[data.pop('split')]
            r = {k: int(v) for k, v in data.items()}
            rep, depth, n = (r[k] for k in ('repeat', 'position', 'suffix'))
            assert 0 <= rep < repeats and (depth, n, tier) in SHAPES and tag in ARM and fit == (depth == 512)
            assert tag not in samples[depth, n, tier, rep] and r['bytes'] == (16777216 if tier == 'RAM' else 52690944)
            assert 0 <= r['order'] <= 3 and r['qpc_hz'] > 0
            c, x = tuple(r['compute_' + a] for a in ('start', 'submitted', 'ready')), tuple(r['transfer_' + a] for a in ('start', 'submitted', 'ready'))
            if tag == 'transfer-only':
                assert c == (-1, -1, -1) and r['compute_timeline'] == 0
            else:
                assert 0 <= c[0] <= c[1] < c[2] and r['compute_timeline'] > 0
            if tag.startswith('compute'):
                assert x == (-1, -1, -1) and r['dma_ticket'] == 0
            else:
                assert 0 <= x[0] <= x[1] < x[2] and r['dma_ticket'] > 0
            assert bool(r['disk_us']) == (tier == 'SSD' and not tag.startswith('compute'))
            pending = [r['compute_pending_after_dma_' + a] for a in ('submit', 'ready')]
            assert all(p in (0, 1) for p in pending)
            assert tag == 'joint' or pending == [0, 0]
            assert tag != 'joint' or c[1] <= x[0]
            samples[depth, n, tier, rep][tag] = (r, c, x, pending)
    assert set(samples) == {(p, n, t, r) for p, n, t in SHAPES for r in range(repeats)}
    fit_residual, blocks = defaultdict(list), defaultdict(list)
    for (p, n, tier, rep), a in sorted(samples.items()):
        assert set(a) == ARM and {item[0]['order'] for item in a.values()} == set(range(4))
        frequencies = {item[0]['qpc_hz'] for item in a.values()}
        freq, = frequencies
        ms = lambda ticks: Fraction(1000 * ticks, freq)
        (_, ca, _, _), (_, cb, _, _), (_, _, xo, _), (_, c, x, pending) = (a[k] for k in ('compute-a', 'compute-b', 'transfer-only', 'joint'))
        solo_compute = (ms(ca[2] - ca[0]) + ms(cb[2] - cb[0])) / 2
        solo_transfer = ms(xo[2] - xo[0])
        delta = ms(x[0] - c[0])
        naive = max(solo_compute, delta + solo_transfer)
        actual = ms(max(c[2], x[2]) - c[0])
        if p == 512:
            fit_residual[tier].append(actual - naive)
        blocks[p, n, tier].append(dict(solo_compute=solo_compute, solo_transfer=solo_transfer, joint_compute=ms(c[2] - c[0]),
                                         joint_transfer=ms(x[2] - x[0]), ideal=naive, actual=actual,
                                         compute_start_gap=delta, aa=abs(ms(ca[2] - ca[0]) - ms(cb[2] - cb[0])),
                                         pending_at_submit=pending[0], pending_at_ready=pending[1]))
    correction = {tier: median(rows) for tier, rows in fit_residual.items()}
    result = []
    predictions = []
    for (p, n, tier), data in sorted(blocks.items()):
        med = lambda seq: float(median(seq))
        row = dict(position=p, suffix=n, tier=tier, split='fit' if p == 512 else 'validation', samples=len(data),
                   overlap_submit=sum(d['pending_at_submit'] for d in data), overlap_ready=sum(d['pending_at_ready'] for d in data),
                   solo_compute_ms=med(d['solo_compute'] for d in data), solo_transfer_ms=med(d['solo_transfer'] for d in data),
                   paired_compute_change_ms=med(d['joint_compute'] - d['solo_compute'] for d in data),
                   paired_transfer_change_ms=med(d['joint_transfer'] - d['solo_transfer'] for d in data),
                   paired_makespan_residual_ms=med(d['actual'] - d['ideal'] for d in data),
                   aa_compute_p90_ms=float(quantiles([d['aa'] for d in data], n=10, method='inclusive')[-1]))
        if p != 512:
            fixed = [d['ideal'] + correction[tier] for d in data]
            naive = [d['ideal'] for d in data]
            actual = [d['actual'] for d in data]
            row.update(uncorrected_median_abs_error_ms=med(abs(x - y) for x, y in zip(actual, naive, strict=True)),
                       corrected_median_abs_error_ms=med(abs(x - y) for x, y in zip(actual, fixed, strict=True)),
                       uncorrected_max_abs_error_ms=float(max(abs(x - y) for x, y in zip(actual, naive, strict=True))),
                       corrected_max_abs_error_ms=float(max(abs(x - y) for x, y in zip(actual, fixed, strict=True))))
            predictions.append((median(naive), median(fixed), median(actual)))
        result.append(row)
    rank_naive, rank_fixed, comparable = 0, 0, 0
    for a, b in combinations(predictions, 2):
        want = a[-1] - b[-1]
        if want:
            comparable += 1
            rank_naive += (a[0] - b[0]) * want < 0
            rank_fixed += (a[1] - b[1]) * want < 0
    resource = {}
    for line in stderr.read_text(encoding='utf8', errors='replace').splitlines():
        if line.startswith('r10_resources,'):
            _, phase, *nums = line.split(',')
            resource[phase] = dict(zip(('device_free', 'device_budget', 'process_working_set', 'peak_working_set', 'process_private_commit'),
                                       map(int, nums), strict=True))
    assert set(resource) == {'before-model', 'after-model', 'after-context', 'after-grid-preparation', 'after-timed-grid'}
    return dict(scope='actual native-device fence ordering and paired service-path phase timings; not a strategy experiment',
                cases=result, observations=4 * len(samples), model_correction_ms={k: float(v) for k, v in correction.items()},
                validation_comparable_pairs=comparable, validation_rank_errors=dict(ideal=rank_naive, corrected=rank_fixed),
                resource_snapshots=resource)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ('input', 'stderr', 'receipt', 'archive', 'out'):
        ap.add_argument('--' + arg, required=True, type=Path)
    a = ap.parse_args()
    if a.out.exists() or a.archive.exists():
        raise FileExistsError('choose new evidence paths')
    receipt = json.loads(a.receipt.read_bytes())
    assert receipt['mode'] == 'joint' and receipt['return_code'] == 0 and receipt['repeats'] >= 12
    report = score(a.input, a.stderr, receipt['repeats'])
    with ZipFile(a.archive, 'x') as z:
        for p in (a.input, a.stderr, a.receipt):
            z.writestr(ZipInfo(p.name), p.read_bytes(), compress_type=ZIP_DEFLATED)
    report['archive_sha256'] = digest(a.archive)
    report['receipt_sha256'] = digest(a.receipt)
    with a.out.open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), file=f)
    print(f"Audited {report['observations']} matched native timing records.")


if __name__ == '__main__':
    main()
