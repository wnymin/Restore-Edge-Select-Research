"""Build the R10 local execution freeze from complete, calibrated A/A controls."""

import argparse
from fractions import Fraction
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
from zipfile import ZipFile

from scripts.gpu.analysis.summarize_r10_pilot_aa import endpoints, pair
from scripts.gpu.analysis.summarize_r10_pilot_load import measured_window
from scripts.gpu.experiments import r10_holdout_lock as lock
from scripts.validation.check_r10_native_catalog import digest


def quantile(xs, q):
    return sorted(xs)[ceil(len(xs) * q) - 1]


def control(first, second, schedule, worker_hash, observer_hash, pilot_hash, capacity):
    paths = [Path(p) for p in (first, second)]
    schedule = Path(schedule)
    stats = pair(*paths, schedule)
    trace = json.loads(schedule.read_bytes())['trace']
    planned = sorted(Fraction(str(r['arrival_ms'])) for r in trace['requests'])
    span = planned[-1] - planned[0]
    runs = [json.loads(endpoints(path)[0].read_bytes()) for path in paths]
    receipt = runs[0]
    assert span > 0 and all(
        r['worker_sha256'] == worker_hash and r['observer_sha256'] == observer_hash and
        r['cost_profile_sha256'] == pilot_hash and not r.get('holdout_lock_sha256') and
        r['capacity_bytes'] == capacity and r['warmup_grid'] and not r['resources'] and not r['chunk_events'] and
        not r['pipeline_events'] and not r.get('ssd_hold_ms') and r.get('monitor_exit_code') is None for r in runs)
    offsets = list(map(Fraction, receipt['planned_arrival_offsets_ms']))
    rate = offsets[-1] / span
    assert rate > 0 and offsets == [(x - planned[0]) * rate for x in planned]
    windows, responses = [], []
    for r in runs:
        a, z = (list(map(Fraction, r[k])) for k in ('client_arrivals_ms', 'completions_ms'))
        windows.append(measured_window(a, z))
        responses.append([v - u for u, v in zip(a, z, strict=True)])
    drift = float(quantile([abs(x - y) for x, y in zip(*responses, strict=True)], .95))
    return dict(block=receipt['block'], rate=str(rate), schedule_sha256=digest(schedule), windows=windows,
                receipt_sha256=[digest(endpoints(path)[0]) for path in paths],
                native_event_sha256=[digest(endpoints(path)[1]) for path in paths],
                abs_mean_difference_ms=stats['abs_mean_difference_ms'], abs_p95_difference_ms=stats['abs_p95_difference_ms'],
                p95_paired_request_drift_ms=drift)


def encoded(row):
    return (json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('ascii')


def review_gates(raw, capacity):
    paths = {key: Path(path) for key, path in raw}
    data = {key: json.loads(p.read_bytes()) for key, p in paths.items()}
    interface, active, logging, natural, controlled, replication = (data[k] for k in (
        'interface', 'active-resources', 'logging-perturbation', 'mechanism-native', 'mechanism-controlled',
        'mechanism-replications'))
    if not (interface['full_state_matches'] == interface['final_logits_matches'] >= 8 and len(interface['fixtures']) >= 3):
        raise ValueError('native mixed-path qualification is incomplete')
    budgets = {64 * 2**20, capacity, 144 * 2**20}
    if (len(active['arms']) != 3 or {r['capacity_bytes'] for r in active['arms']} != budgets or
            len({r['block'] for r in active['arms']}) != 1 or not all(
                r.get('active_gpu_samples', 0) >= 2 and r.get('active_system_samples', 0) >= 2 and
                r['event_audit']['requests'] > 0 for r in active['arms'])):
        raise ValueError('the three resource budgets require timed-window GPU and system memory samples on one block')
    if len(logging['records']) != 2 or logging['mean_change_reaches_reference']:
        raise ValueError('logging must be measured below the predeclared effect reference before freezing')
    if natural.get('soft_ssd_hold_ms', 0) or controlled['soft_ssd_hold_ms'] <= 0 or any(
            natural[k] != controlled[k] for k in ('schedule_sha256', 'capacity_bytes')) or any(
            natural[a]['audit']['runtime_worker_sha256'] != controlled[a]['audit']['runtime_worker_sha256'] for a in ('shallow', 'deep')):
        raise ValueError('mechanism gate requires the same input and binary in native and controlled arms')
    pairs = replication['pairs']
    if (set(pairs) != {'repeat-shared-tight', 'repeat-shared-soft700', 'repeat-shared-wide', 'repeat-private-tight',
                      'short-head-tight', 'cached-head-inflight', 'cached-head-compute'} or
            pairs['repeat-shared-tight']['schedule_sha256'] == natural['schedule_sha256']):
        raise ValueError('independent prompt, predecessor length and in-flight mechanism controls must be audited')
    for key, p in paths.items():
        if not p.name.endswith('-summary.json'):
            raise ValueError(f'{key}: evidence summary must name its adjacent archive')
        archive = p.with_name(p.name.removesuffix('-summary.json') + '.zip')
        if digest(archive) != data[key]['archive_sha256']:
            raise ValueError(f'{key}: evidence archive hash does not match')
        with ZipFile(archive) as z:
            if not z.namelist() or z.testzip() is not None:
                raise ValueError(f'{key}: evidence archive is empty or corrupt')
    return {key: dict(name=p.name, sha256=digest(p), archive_sha256=data[key]['archive_sha256']) for key, p in paths.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('layout', 'manifest', 'worker', 'observer', 'probe', 'model', 'runtime', 'pilot-cost', 'cost-out', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--aa', action='append', nargs=4, metavar=('SOURCE', 'FIRST_PREFIX', 'SECOND_PREFIX', 'SCHEDULE'), required=True)
    ap.add_argument('--gate', action='append', nargs=2, metavar=('NAME', 'EVIDENCE'), required=True)
    ap.add_argument('--device', required=True)
    ap.add_argument('--device-index', type=int, required=True)
    ap.add_argument('--timeout-s', type=int, required=True)
    a = ap.parse_args()
    if a.cost_out.exists() or a.out.exists() or a.cost_out.resolve() == a.out.resolve():
        raise FileExistsError('the execution freeze and its cost profile require fresh paths')
    # Do not consult held-out request content when computing scales or thresholds.
    layout = json.loads(a.layout.read_bytes())
    if layout['scope'] != 'membership and pair ordering exclusively; not a holdout execution authorization' or \
            digest(a.manifest) != layout['source_manifests']['native_sha256']:
        raise ValueError('holdout layout is not bound to this manifest')
    sources = {r['source'] if r['family'] == 'bailian' else 'oasst1' for r in layout['blocks']}
    if sorted(src for src, *_ in a.aa) != sorted(sources):
        raise ValueError('supply one complete clean baseline A/A for every registered source')
    required = {'interface', 'active-resources', 'logging-perturbation', 'mechanism-native', 'mechanism-controlled',
                'mechanism-replications'}
    if {k for k, _ in a.gate} != required or len(a.gate) != len(required):
        raise ValueError('supply the reviewed evidence for each method gate')
    model_hash, worker_hash, observer_hash, pilot_hash = map(digest, (a.model, a.worker, a.observer, a.pilot_cost))
    capacity = Fraction('82.25')
    capacity_bytes = int(capacity * (1 << 20))
    gates = review_gates(a.gate, capacity_bytes)
    controls = {s: control(*rest, worker_hash, observer_hash, pilot_hash, capacity_bytes) for s, *rest in a.aa}
    manifest = {r['id']: r for r in json.loads(a.manifest.read_bytes())['blocks']}
    for s, r in controls.items():
        m = manifest[r['block']]
        assert m['schedule_sha256'] == r['schedule_sha256'] and m['split'] in ('pilot', 'calibration')
        expected = 'oasst1' if m['family'] == 'oasst' else r['block'].rsplit('-', 2)[0] + '.jsonl'
        assert expected == s
        assert all(a.timeout_s * 1000 > 1.5 * w['drain_elapsed_ms'] for w in r['windows'])
        if not all(w['fraction_time_any_request_outstanding'] >= .65 and w['max_outstanding'] >= 2 and
                   w['drain_after_last_arrival_ms'] <= .12 * w['submission_span_ms'] for w in r['windows']):
            raise ValueError(f'{s}: the paired baseline does not meet the registered contention window')
    pilot_cost = json.loads(a.pilot_cost.read_bytes())
    assert pilot_cost['scope'] == 'R10 provisional isolated costs for calibration only' and pilot_cost['model_sha256'] == model_hash
    cost = dict(pilot_cost, scope='R10 frozen stationary costs for primary holdout',
                calibration_cost_sha256=pilot_hash, overlap='No fixed overlap correction fitted; no omission-loss claim.')
    frozen_cost = encoded(cost)
    frozen = dict(scope=lock.SCOPE, layout_sha256=digest(a.layout), native_manifest_sha256=digest(a.manifest),
                  blocks=layout['blocks'], run_pairs=layout['run_pairs'], cost_profile_sha256=sha256(frozen_cost).hexdigest(),
                  environment=dict(device=a.device, device_index=a.device_index, model_sha256=model_hash,
                                   runtime_sha256=lock.runtime_hashes(a.runtime), catalogue_probe_sha256=digest(a.probe),
                                   worker_sha256=worker_hash, observer_sha256=observer_hash,
                                   controller_sha256=lock.controller_hashes(), analysis_sha256=lock.analysis_hashes()),
                  calibration_aa=controls, reviewed_gates=gates,
                  timing=dict(capacity_mib=str(capacity), arrival_scale={s: row['rate'] for s, row in controls.items()}, timeout_s=a.timeout_s),
                  acceptance=dict(min_absolute_gain_ms=5, min_relative_gain=.03,
                                  aa_p90_ms=quantile([r['abs_mean_difference_ms'] for r in controls.values()], .90),
                                  p95_absolute_noninferiority_ms=10, p95_relative_noninferiority=.05,
                                  request_wait_budget_ms=max(100, ceil(max(r['p95_paired_request_drift_ms'] for r in controls.values()))),
                                  max_fraction_over_wait_budget=.05, max_failed_request_fraction_increase=0))
    with a.cost_out.open('xb') as f:
        f.write(frozen_cost)
    with a.out.open('xb') as f:
        f.write(encoded(frozen))
    lock.load(a.out)


if __name__ == '__main__':
    main()
