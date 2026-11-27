"""Verify isolated native transfer counters and summarize phase durations."""
import argparse
from collections import defaultdict
import csv
from itertools import combinations
import json
from pathlib import Path
from statistics import median
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from scripts.validation.check_r10_native_catalog import digest


SIZES = {('RAM', x * (1 << 20)) for x in (8, 16, 32)} | {('SSD', 52690944)}


def score(path, repeats):
    buckets = defaultdict(dict)
    order_seen = defaultdict(set)
    with path.open(encoding='ascii', newline='') as f:
        for row in csv.DictReader(f):
            tier = row.pop('tier')
            row = {k: int(v) for k, v in row.items()}
            rep, order, n = (row[k] for k in ('repeat', 'order', 'bytes'))
            assert 0 <= rep < repeats and 0 <= order < len(SIZES) and (tier, n) in SIZES
            assert order not in order_seen[rep] and rep not in buckets[tier, n] and row['exact'] == 1
            disk, host, upload, total, scatter = (row[k] for k in ('disk_us', 'host_to_submit_us', 'submit_to_visible_us',
                                                                  'end_to_end_us', 'device_scatter_us'))
            assert 0 <= disk <= host and upload > 0 and scatter > 0 and host + upload == total
            assert (row['process_read_bytes'] == 0 and disk == 0 if tier == 'RAM' else row['process_read_bytes'] == n and disk > 0)
            buckets[tier, n][rep] = dict(disk=disk, host_call_excluding_disk=host - disk, upload_signal=upload,
                                        end_to_end=total, device_scatter_16_spans=scatter)
            order_seen[rep].add(order)
    assert set(buckets) == SIZES and all(len(x) == repeats for x in buckets.values())
    summaries = []
    for (kind, n), samples in sorted(buckets.items()):
        subsets = {}
        for field in samples[0]:
            values = [x[field] / 1000 for x in samples.values()]
            subsets[field] = dict(median_ms=median(values), min_ms=min(values), max_ms=max(values))
        summaries.append(dict(tier=kind, bytes=n, phases=subsets, repetitions=len(samples)))
    rates = [(r['bytes'], r['phases']['end_to_end']['median_ms']) for r in summaries if r['tier'] == 'RAM']
    slope = median((y1 - y0) / (x1 - x0) for (x0, y0), (x1, y1) in combinations(rates, 2))
    assert slope > 0
    startup = max(0, median(t - slope * n for n, t in rates))
    return dict(scope='isolated native transfer lane; scatter into 16 contiguous spans of a device-only target; provisional',
                records=len(SIZES) * repeats, phase_costs=summaries,
                repeated_unbuffered_read_bytes=repeats * 52690944,
                provisional_ram_fit=dict(startup_ms=startup, bytes_per_ms=1 / slope))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'receipt', 'stderr', 'archive', 'out'):
        ap.add_argument('--' + name, required=True, type=Path)
    a = ap.parse_args()
    if a.archive.exists() or a.out.exists():
        raise FileExistsError('select new evidence paths')
    receipt = json.loads(a.receipt.read_bytes())
    assert receipt['return_code'] == 0 and receipt['repeats'] >= 12
    summary = score(a.input, receipt['repeats'])
    with ZipFile(a.archive, 'x') as z:
        for path in (a.input, a.receipt, a.stderr):
            z.writestr(ZipInfo(path.name), path.read_bytes(), compress_type=ZIP_DEFLATED)
    summary['archive_sha256'] = digest(a.archive)
    with a.out.open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2), file=f)
    print(f"Verified {summary['records']} isolated transfers and exact device readbacks.")


if __name__ == '__main__':
    main()
