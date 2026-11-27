"""Run the registered local-cost rate ladder on sealed calibration inputs only."""

import argparse
from fractions import Fraction
import json
from math import ceil
from pathlib import Path

from scripts.gpu.experiments import r10_holdout_lock
from scripts.gpu.experiments.run_r10_catalog_prepare import write_new
from scripts.gpu.experiments.run_r10_local_method_checks import QueueBoundary, plan, timed
from scripts.validation.check_r10_native_catalog import digest


SCOPE = 'R10 single-slot HTTP/SSE service calibration; no held-out execution'
EXTENSION_SCOPE = 'R10 additional service rates from baseline calibration only'


def matrix(design, added=()):
    expected = ('block', 'profile', 'rates')
    maps = ('blocks', 'output_profiles', None)
    orders = []
    for name, mapping in zip(expected, maps, strict=True):
        order = design[name + '_order']
        if len(order) != len(set(order)) or set(order) != (set(design[mapping]) if mapping else {'low', 'middle', 'high'}):
            raise ValueError(f'invalid calibration order: {name}')
        orders.append(order)
    if len(orders[0]) != 2 or set(orders[1]) != {'short', 'mixed'}:
        raise ValueError('expected two calibration inputs and two output profiles')
    for block in orders[0]:
        details = design['blocks'][block]
        scale_low, scale_middle, scale_high = map(Fraction, (details['arrival_scale'][k] for k in ('low', 'middle', 'high')))
        if not scale_low > scale_middle > scale_high > 0:
            raise ValueError(f'{block}: high rate must have the smallest clock multiplier')
        for profile in orders[1]:
            setting = design['output_profiles'][profile]
            for rate in orders[2]:
                yield block, profile, rate, details, setting
    used = set()
    for row in added:
        block, profile, rate, scale = (row[k] for k in ('block', 'profile', 'rate', 'arrival_scale'))
        details = design['blocks'][block]
        setting = design['output_profiles'][profile]
        key = block, profile, rate
        scale = Fraction(scale)
        if (key in used or rate in details['arrival_scale'] or not rate.startswith('extra') or scale <= 0 or
                scale in map(Fraction, details['arrival_scale'].values())):
            raise ValueError(f'invalid added baseline rate: {row}')
        used.add(key)
        yield block, profile, rate, dict(details, arrival_scale={rate: str(scale)}), setting


def extensions(paths, design_sha256):
    added = []
    seen = set()
    for p in paths:
        d = json.loads(p.read_bytes())
        if (d['scope'] != EXTENSION_SCOPE or d['design_sha256'] != design_sha256 or not d['points'] or
                any(len(d[k]) != 64 for k in ('baseline_summary_sha256', 'baseline_archive_sha256'))):
            raise ValueError(f'incompatible service rate extension: {p}')
        for row in d['points']:
            key = row['block'], row['profile'], Fraction(row['arrival_scale'])
            if key in seen:
                raise ValueError(f'added baseline clock already registered: {key}')
            seen.add(key)
        added.extend(d['points'])
    return added


def service_deadline(design, source_rows, scale):
    clocks = [Fraction(str(r['arrival_ms'])) for r in source_rows]
    if not clocks or clocks != sorted(clocks):
        raise ValueError('expected nonempty ordered arrival window')
    span = (clocks[-1] - clocks[0]) * Fraction(scale) / 1000
    deadline = max(design['timeout_s'], ceil(span + max(Fraction(60), span / 4)))
    if deadline > design['extension_rule']['hard_timeout_s']:
        raise ValueError(f'planned span {float(span):.1f}s exceeds registered service window guard')
    return deadline


def generation_shape(schedule, output):
    context = schedule['model']['context_tokens']
    short, long = output['short_tokens'], output['long_tokens']
    if not (0 < short <= 256 and (long == 0 or short < long <= 256 and output['long_request_index_modulo'] == 4)):
        raise ValueError('output profile differs from the native every-fourth-request rule')
    rows = schedule['trace']['requests']
    minimum_suffix, minimum_headroom, maximum_prompt, candidates, long_requests = context, context, 0, 0, 0
    for i, row in enumerate(rows):
        cap = long if long and i % 4 == 0 else short
        length = sum(part['count'] for part in row['input_token_runs'])
        cut = min(length - q['boundary_tokens'] for q in row['candidates'])
        room = context - length - cap
        if cut < 1 or room < 0:
            raise ValueError(f'request {i} cannot produce its first token or exceeds the native decode context')
        minimum_suffix, minimum_headroom = min(minimum_suffix, cut), min(minimum_headroom, room)
        maximum_prompt = max(maximum_prompt, length)
        candidates += len(row['candidates'])
        long_requests += bool(long and cap == long)
    if not rows:
        raise ValueError('service block has no requests')
    return dict(requests=len(rows), all_candidates=candidates, context_tokens=context, longest_prompt_tokens=maximum_prompt,
                minimum_prompt_suffix_tokens=minimum_suffix, minimum_post_generation_context_headroom=minimum_headroom,
                long_cap_requests=long_requests)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('design', 'root', 'worker', 'runtime', 'model', 'observer', 'cost-profile', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--device', default='Vulkan0')
    ap.add_argument('--device-index', type=int, default=1)
    ap.add_argument('--max-new-steps', type=int, default=1)
    ap.add_argument('--extension', type=Path, action='append', default=[])
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    a.started = 0
    if a.max_new_steps < 0:
        ap.error('--max-new-steps must be nonnegative')
    design = json.loads(a.design.read_bytes())
    added = extensions(a.extension, digest(a.design))
    if design['scope'] != SCOPE or design['policy'] != 'local-cost':
        raise ValueError('the service ladder is calibration using only the baseline policy')
    if not a.dry_run:
        a.output.mkdir(parents=True, exist_ok=True)
        schema_inputs = ('worker', 'observer', 'model', 'cost_profile')
        identity = dict(design_sha256=digest(a.design), device=a.device, device_index=a.device_index,
                        sha256={k: digest(getattr(a, k)) for k in schema_inputs},
                        controller_sha256=r10_holdout_lock.controller_hashes(),
                        service_sha256={name: digest(Path(__file__).with_name(name)) for name in
                                        ('r10_service_events.py', 'r10_service_sse.py', 'r10_service_lock.py', 'run_r10_service_calibration.py')},
                        runtime_sha256=r10_holdout_lock.runtime_hashes(a.runtime))
        mark = a.output / 'design.json'
        if not mark.exists():
            write_new(mark, identity)
        elif json.loads(mark.read_bytes()) != identity:
            raise ValueError('service calibration directory already belongs to another design or executable')

    plans = a.root / 'native-plans'
    for block, profile, rate, details, setting in matrix(design, added):
        _, schedule = plan(plans, block)
        if digest(schedule) != details['schedule_sha256']:
            raise ValueError(f'calibration schedule changed: {block}')
        shape = json.loads(schedule.read_bytes())
        generation_shape(shape, setting)
        catalogue = a.root / 'native-catalogs' / details['catalogue']
        label = a.output / f'{block}-{profile}-{rate}'
        failure = label.with_suffix('.failure.json')
        if failure.exists() and not label.with_suffix('.run.json').exists() and not a.dry_run:
            note = json.loads(failure.read_bytes())
            if (note['block'], note['policy']) != (block, design['policy']):
                raise ValueError(f'incompatible service failure record: {failure}')
            print(f'PRESERVE failed calibration arm {label.name}', flush=True)
            continue
        options = dict(service_tokens=setting['short_tokens'], service_long_tokens=setting['long_tokens'],
                       service_sse=True, timeout_s=service_deadline(design, shape['trace']['requests'], details['arrival_scale'][rate]))
        timed(a, label, a.worker, plans, block, catalogue, design['capacity_mib'], details['arrival_scale'][rate], options)


if __name__ == '__main__':
    try:
        main()
    except QueueBoundary as e:
        print(f'STOP {e}', flush=True)
