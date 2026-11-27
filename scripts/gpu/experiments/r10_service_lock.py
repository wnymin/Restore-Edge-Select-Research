"""Enforce a separate lock for held-out single-slot streaming runs."""

from fractions import Fraction
import json
from pathlib import Path

from scripts.gpu.experiments import r10_holdout_lock as primary
from scripts.validation.check_r10_native_catalog import digest


SCOPE = 'R10 local streaming service holdout execution v1'
PROFILES = ('short', 'mixed')
ANCHORS = ('low', 'shoulder', 'overload')


def source_hashes():
    return {name: digest(Path(__file__).with_name(name)) for name in
            ('r10_service_lock.py', 'r10_service_events.py', 'r10_service_sse.py')}


def analysis_hashes():
    analysis = Path(__file__).resolve().parents[1] / 'analysis'
    return {name: digest(analysis / name) for name in
            ('audit_r10_service_pipe.py', 'summarize_r10_service_holdout.py',
             'summarize_r10_pilot_load.py', 'summarize_r10_pilot_aa.py', 'check_r10_catalog_fifo.py')}


def load(path):
    lock = json.loads(path.read_bytes())
    if lock.get('scope') != SCOPE:
        raise ValueError('held-out service requires a distinct frozen service lock')
    blocks = {x['id']: x for x in lock['blocks']}
    if len(blocks) != len(lock['blocks']) or sorted(x['family'] for x in blocks.values()) != ['bailian'] * 2 + ['oasst'] * 2:
        raise ValueError('service holdout requires two independent blocks per input family')
    cases = lock['cases']
    expected = {(b, p, a) for b in blocks for p in PROFILES for a in ANCHORS}
    actual = {(r['block'], r['profile'], r['anchor']) for r in cases}
    if len(cases) != 24 or len({r['id'] for r in cases}) != len(cases) or actual != expected:
        raise ValueError('service lock requires one paired case per block, profile and rate anchor')
    if Fraction(lock['capacity_mib']) <= 0 or lock['timeout_s'] <= 0 or any(
            sorted(x['arm_order']) != sorted(primary.POLICIES) or Fraction(x['arrival_scale']) <= 0 or
            not 1 <= x['short_tokens'] <= 256 or not (x['long_tokens'] == 0 if x['profile'] == 'short' else
                x['short_tokens'] < x['long_tokens'] <= 256) for x in cases):
        raise ValueError('invalid service arm order, clock multiplier, budget or output cap')
    if not all(0 < v < float('inf') for v in (lock['experimental_slo']['visible_ttft_ms'],
                                              lock['experimental_slo']['client_request_tpot_ms'])):
        raise ValueError('service lock must include positive predeclared SLOs')
    return lock


def check_environment(lock):
    primary.check_controller(lock)
    if source_hashes() != lock['environment']['service_sha256']:
        raise ValueError('streaming service implementation changed since the service holdout freeze')
    if analysis_hashes() != lock['environment']['service_analysis_sha256']:
        raise ValueError('service analysis changed since the service holdout freeze')


def preparation(lock, row, args):
    check_environment(lock)
    if digest(args.manifest) != lock['membership_sha256']:
        raise ValueError('service catalogue preparation must use the frozen membership')
    primary.preparation(lock, row, args)


def timing(lock, path, seal, args):
    check_environment(lock)
    row = primary.block(lock, seal['block'], digest(args.plan), digest(args.schedule))
    cases = [x for x in lock['cases'] if x['id'] == args.service_case and x['block'] == row['id']]
    if (len(cases) != 1 or (seal['split'], seal['family']) != (row['split'], row['family']) or
            seal.get('service_lock_sha256') != digest(path)):
        raise ValueError('service case and prepared catalogue must match the service holdout lock')
    c = cases[0]
    if (args.capacity_mib, args.arrival_scale, args.timeout_s, args.service_tokens, args.service_long_tokens) != (
            Fraction(lock['capacity_mib']), Fraction(c['arrival_scale']), lock['timeout_s'], c['short_tokens'], c['long_tokens']):
        raise ValueError('service capacity, arrival scale, timeout or output caps changed')
    if (args.policy not in c['arm_order'] or args.publish_rest_on != 'open-loop' or not (args.service_sse and args.warmup_grid and args.warmup) or
            any((args.forced_actions, args.allocate_only, args.verify, args.quiet_events, args.resources, args.monitor_wddm,
                 args.chunk_events, args.pipeline_events, args.ssd_hold_ms))):
        raise ValueError('only registered HTTP/SSE service arms with uniform warmup may run')
    if (not args.cost_profile or digest(args.cost_profile) != lock['cost_profile_sha256'] or
            digest(args.worker) != lock['environment']['worker_sha256'] or digest(args.observer) != lock['environment']['observer_sha256']):
        raise ValueError('service worker, observer or cost profile changed')
    primary.model_backend(lock, args.model, args.runtime, args.device, args.device_index)
    return row
