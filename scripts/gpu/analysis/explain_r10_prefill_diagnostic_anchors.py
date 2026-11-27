"""Audit archived native prefill controls and replay both choices on the same states."""

import argparse
from bisect import bisect_left
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from zipfile import ZipFile

from scripts.gpu.analysis.check_r10_catalog_fifo import check
from scripts.gpu.analysis.explain_r10_primary_actions import explain
from scripts.gpu.analysis.summarize_r10_pilot_load import measured_window
from scripts.validation.check_r10_native_catalog import digest


# Each arm was selected by its original experiment and observed load, before chooser replay.
# Fields: condition, archive stem, archive-hash report, load report, recorded arm prefix.
ANCHORS = (
    ('capacity_64_mib', 'active-capacity-sweep', 'active-capacity-sweep-summary', 'active-capacity-sweep-load', 'resource-64'),
    ('capacity_82_25_mib', 'active-capacity-sweep', 'active-capacity-sweep-summary', 'active-capacity-sweep-load', 'resource-8225'),
    ('capacity_144_mib', 'active-capacity-sweep', 'active-capacity-sweep-summary', 'active-capacity-sweep-load', 'resource-144'),
    ('lighter_thinking', 'thinking-load-initial', 'thinking-load-initial', 'thinking-load-initial', 'thinking29-140-first'),
    ('trace_a_dense_arrivals', 'warmed-aa-traceA09', 'warmed-aa-traceA09-summary', 'warmed-load-v1', 'traceA09-a1'),
    ('early_coding_backlog', 'warmup-grid-bailian', 'warmup-grid-bailian-summary', 'warmup-load-initial', 'bailian-coder-28-grid'),
)


class Member:
    def __init__(self, data):
        self.data = data

    def read_bytes(self):
        return self.data

    def open(self, *, encoding, newline):
        return StringIO(self.data.decode(encoding), newline=newline)


def read(path):
    return json.loads(path.read_bytes())


def peak_in_window(arrivals, width_ms):
    return max(bisect_left(arrivals, t + width_ms, lo=i) - i for i, t in enumerate(arrivals))


def arm(evidence, plans, cost, cost_hash, key):
    condition, zip_stem, hash_stem, load_stem, label = key
    path = lambda stem, suffix: evidence / f'r10-native-{stem}{suffix}'
    archive, hash_report, load = path(zip_stem, '.zip'), path(hash_stem, '.json'), path(load_stem, '.json')
    archive_hash = digest(archive)
    if archive_hash != read(hash_report)['archive_sha256']:
        raise ValueError(f'archive changed: {archive}')
    expected, = [r for r in read(load)['arms'] if r['run'] == label]
    schedule = plans / (expected['block'] + '.schedule.json')
    if digest(schedule) != expected['source_schedule_sha256']:
        raise ValueError(f'schedule changed: {schedule}')
    with ZipFile(archive) as z:
        names = [f'{label}.{suffix}' for suffix in ('run.json', 'native.csv')]
        if any(z.namelist().count(name) != 1 for name in names):
            raise ValueError(f'missing or duplicate archive members: {label}')
        receipt, native = (Member(z.read(name)) for name in names)
    recorded = read(receipt)
    receipt_hash = sha256(receipt.read_bytes()).hexdigest()
    if (receipt_hash, recorded['block'], recorded['capacity_bytes'], recorded['cost_profile_sha256'],
            recorded['policy'], recorded.get('ssd_hold_ms', 0), recorded['warmup_grid']) != (
            expected['receipt_sha256'], expected['block'], expected['capacity_bytes'], cost_hash, 'local-cost', 0, True):
        raise ValueError(f'run does not match the original prefill diagnostic: {label}')
    arrivals, finishes = (list(map(Fraction, recorded[field])) for field in ('client_arrivals_ms', 'completions_ms'))
    window = measured_window(arrivals, finishes)
    if window != expected['window']:
        raise ValueError(f'load metrics did not reproduce: {label}')
    audit = check(receipt, native, schedule)
    paired = explain(recorded, read(schedule), cost)
    del paired['chooser_called_after_snapshot_ms']
    snapshots = [r['snapshot'] for r in recorded['decisions']]
    return dict(condition=condition, run=label, block=recorded['block'], capacity_bytes=recorded['capacity_bytes'],
                inputs=dict(archive=archive.name, archive_sha256=archive_hash, archive_hash_report=hash_report.name,
                            archive_hash_report_sha256=digest(hash_report), load_report=load.name, load_report_sha256=digest(load),
                            receipt_sha256=receipt_hash, native_csv_sha256=sha256(native.read_bytes()).hexdigest(),
                            schedule_sha256=expected['source_schedule_sha256']),
                load=window, peak_client_submissions_in_100_ms=peak_in_window(arrivals, 100),
                capture=dict(chunk_events=recorded['chunk_events'], pipeline_events=recorded['pipeline_events'],
                             peak_prefetch_bytes=audit['peak_prefetch_bytes']),
                snapshots_with_active_compute=sum(s['compute_start_ms'] is not None for s in snapshots),
                snapshots_with_transfer_in_flight=sum(s['transfer_object'] is not None for s in snapshots),
                device_event_counts=audit['device_event_counts'], **paired)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('evidence', 'plans', 'cost-profile', 'out'):
        p.add_argument('--' + flag, type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    cost, cost_hash = read(a.cost_profile)['cost'], digest(a.cost_profile)
    rows = [arm(a.evidence, a.plans, cost, cost_hash, key) for key in ANCHORS]
    counts = Counter()
    for r in rows:
        counts.update(r['counts'])
    result = dict(scope='archived baseline prefill snapshots, same-state choice only; no closed-loop or timing comparison',
                  peak_client_window='maximum submissions in any half-open 100 ms interval; no injected burst',
                  cost_profile_sha256=cost_hash, independent_source_blocks=len({r['block'] for r in rows}),
                  counts=dict(counts), arms=rows)
    with a.out.open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(result, indent=2, sort_keys=True), file=f)
    print(f"{len(rows)} native arms, {counts['snapshots']} decisions, {counts['same_snapshot_disagreements']} changes.")


if __name__ == '__main__':
    main()
