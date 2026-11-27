"""Guards separating R10 calibration, catalogue preparation, and held-out timing."""

import json
from fractions import Fraction
from pathlib import Path

from scripts.validation.check_r10_native_catalog import digest


SCOPE = 'R10 local primary holdout execution v1'
POLICIES = ('local-cost', 'entry-replay')
ROOT = Path(__file__).resolve().parents[3]
CONTROLLER = (
    'src/restore_edge_select/adaptive.py',
    'src/restore_edge_select/live_fifo.py',
    'src/restore_edge_select/live_snapshot.py',
    'src/restore_edge_select/prefill_cost.py',
    'src/restore_edge_select/simulator.py',
    'scripts/gpu/experiments/r10_holdout_lock.py',
    'scripts/gpu/experiments/run_r10_catalog_fifo.py',
    'scripts/gpu/experiments/run_r10_catalog_prepare.py',
    'scripts/gpu/experiments/run_r10_fifo_link.py',
    'scripts/gpu/experiments/run_r10_holdout_batch.py',
    'scripts/gpu/experiments/run_r10_online_diagnostic.py',
)
ANALYSIS = ('scripts/gpu/analysis/check_r10_catalog_fifo.py', 'scripts/gpu/analysis/summarize_r10_holdout.py')


def controller_hashes():
    return {name: digest(ROOT / name) for name in CONTROLLER}


def analysis_hashes():
    return {name: digest(ROOT / name) for name in ANALYSIS}


def check_controller(lock):
    if controller_hashes() != lock['environment']['controller_sha256']:
        raise ValueError('Python controller or chooser changed since the holdout freeze')


def runtime_hashes(directory):
    return {p.name: digest(p) for p in sorted(directory.glob('*.dll'))}


def load(path):
    lock = json.loads(path.read_bytes())
    if lock.get('scope') != SCOPE:
        raise ValueError('held-out execution requires an explicit frozen run lock')
    all_blocks = lock['blocks']
    b = {r['id']: r for r in all_blocks}
    pairs = lock['run_pairs']
    if len(all_blocks) != len(b) or len(b) != 12 or {x['family'] for x in b.values()} != {'oasst', 'bailian'} or any(
            sum(r['family'] == q for r in b.values()) != 6 for q in {'oasst', 'bailian'}):
        raise ValueError('holdout lock requires six different blocks per family')
    if len(pairs) != 60 or {(r['block'], r['round']) for r in pairs} != {(block, k) for block in b for k in range(1, 6)} or any(
            sorted(r['arm_order']) != sorted(POLICIES) for r in pairs):
        raise ValueError('holdout lock requires five complete, ordered pairs per block')
    expected_sources = {r['source'] if r['family'] == 'bailian' else 'oasst1' for r in b.values()}
    timing = lock['timing']
    if set(timing['arrival_scale']) != expected_sources or any(Fraction(v) <= 0 for v in timing['arrival_scale'].values()) or \
            timing['timeout_s'] <= 0 or Fraction(timing['capacity_mib']) <= 0:
        raise ValueError('holdout lock requires a positive timing configuration for every source')
    acceptance = lock['acceptance']
    for name in ('min_absolute_gain_ms', 'p95_absolute_noninferiority_ms', 'request_wait_budget_ms', 'aa_p90_ms'):
        if not 0 <= acceptance[name] < float('inf'):
            raise ValueError(f'invalid acceptance bound: {name}')
    for name in ('min_relative_gain', 'p95_relative_noninferiority', 'max_fraction_over_wait_budget', 'max_failed_request_fraction_increase'):
        if not 0 <= acceptance[name] <= 1:
            raise ValueError(f'invalid acceptance rate: {name}')
    if set(lock['environment']['controller_sha256']) != set(CONTROLLER):
        raise ValueError('holdout lock does not record every controller dependency')
    if set(lock['environment']['analysis_sha256']) != set(ANALYSIS):
        raise ValueError('holdout lock does not record both primary analysis scripts')
    return lock


def block(lock, ident, wire_digest, schedule_digest):
    matches = [r for r in lock['blocks'] if r['id'] == ident]
    if len(matches) != 1 or (matches[0]['wire_sha256'], matches[0]['schedule_sha256']) != (wire_digest, schedule_digest):
        raise ValueError('holdout input does not match the locked primary block')
    return matches[0]


def arrival_scale(lock, row):
    """Apply only a preregistered source scale; never infer it on holdout."""
    source = row['source'] if row['family'] == 'bailian' else 'oasst1'
    return lock['timing']['arrival_scale'][source]


def model_backend(lock, model, runtime, device, device_index):
    fixed = lock['environment']
    if (device, device_index) != (fixed['device'], fixed['device_index']) or digest(model) != fixed['model_sha256'] or \
            runtime_hashes(runtime) != fixed['runtime_sha256']:
        raise ValueError('model, device or backend do not match the holdout lock')


def preparation(lock, row, args):
    check_controller(lock)
    checked = block(lock, row['id'], row['wire_sha256'], row['schedule_sha256'])
    if row['split'] != checked['split'] or row['family'] != checked['family']:
        raise ValueError('preparation identity differs from the holdout lock')
    if digest(args.probe) != lock['environment']['catalogue_probe_sha256']:
        raise ValueError('catalogue probe changed')
    model_backend(lock, args.model, args.runtime, args.device_name, args.device_index)


def timing(lock, lock_path, seal, args):
    check_controller(lock)
    row = block(lock, seal['block'], digest(args.plan), digest(args.schedule))
    if (seal['split'], seal['family']) != (row['split'], row['family']) or seal.get('holdout_lock_sha256') != digest(lock_path):
        raise ValueError('held-out catalogue must be prepared under this holdout lock')
    if args.round not in range(1, 6) or args.policy not in POLICIES or args.forced_actions or args.allocate_only or args.verify:
        raise ValueError('only registered main arms are timed here')
    if args.publish_rest_on != 'open-loop' or not (args.warmup and args.warmup_grid) or args.quiet_events or \
            args.resources or args.monitor_wddm or args.chunk_events or args.pipeline_events or args.ssd_hold_ms:
        raise ValueError('holdout requires the registered warmup and common event logging')
    fixed = lock['timing']
    if (args.capacity_mib, args.arrival_scale, args.timeout_s) != (Fraction(fixed['capacity_mib']),
            Fraction(arrival_scale(lock, row)), fixed['timeout_s']):
        raise ValueError('timing parameters differ from holdout lock')
    if not args.cost_profile or digest(args.cost_profile) != lock['cost_profile_sha256'] or \
            digest(args.worker) != lock['environment']['worker_sha256'] or \
            digest(args.observer) != lock['environment']['observer_sha256']:
        raise ValueError('worker, observer or cost profile differ from holdout lock')
    model_backend(lock, args.model, args.runtime, args.device, args.device_index)
    return row
