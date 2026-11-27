"""Run the registered R10 method controls serially, without touching holdout data."""

import argparse
import json
from pathlib import Path
from subprocess import CREATE_NO_WINDOW, run
import sys

from scripts.gpu.experiments.run_r10_catalog_prepare import write_new
from scripts.validation.check_r10_native_catalog import digest


class QueueBoundary(Exception):
    """An operator limit reached before a new command may start."""


def launch(a, mod, options, log):
    arguments = [sys.executable, '-m', 'scripts.gpu.experiments.' + mod]
    for key, value in options.items():
        flag = '--' + key.replace('_', '-')
        arguments += ([flag] if value else []) if isinstance(value, bool) else [flag, str(value)]
    if a.dry_run:
        print(json.dumps(dict(module=mod, arguments=arguments[3:]), ensure_ascii=False))
        return
    if (a.output / 'stop.request').exists():
        raise QueueBoundary('stop.request present; no new command started')
    if a.max_new_steps is not None and a.started >= a.max_new_steps:
        raise QueueBoundary(f'reached --max-new-steps ({a.max_new_steps}); no new command started')
    print(f'START {log.stem.removesuffix(".driver")}', flush=True)
    a.started += 1
    with log.open('xb') as f:
        status = run(arguments, stdout=f, stderr=f, creationflags=CREATE_NO_WINDOW, check=False)
    if status.returncode:
        raise RuntimeError(f'{mod} exited {status.returncode}; see {log}')


def plan(root, block):
    return root / f'{block}.plan', root / f'{block}.schedule.json'


def timed(a, prefix, worker, plans, block, catalogue, capacity, scale, plus=None):
    wire, schedule = plan(plans, block)
    seal = catalogue.with_suffix('.seal.json')
    if seal.exists():
        d = json.loads(seal.read_bytes())
        if d['split'] not in ('pilot', 'calibration') or d['block'] != block or digest(schedule) != d['schedule_sha256']:
            raise ValueError(f'unexpected supplied catalogue: {seal}')
    elif not a.dry_run:
        raise FileNotFoundError(seal)
    options = dict(worker=worker, runtime=a.runtime, model=a.model, observer=a.observer, policy='local-cost',
                   out=prefix, plan=wire, schedule=schedule, catalogue=catalogue, seal=seal, cost_profile=a.cost_profile,
                   capacity_mib=capacity, arrival_scale=scale, timeout_s=280, warmup=True, warmup_grid=True,
                   device=a.device, device_index=a.device_index)
    options.update(plus or {})
    if prefix.with_suffix('.run.json').exists() and not a.dry_run:
        print(f'PRESERVE completed {prefix.name}', flush=True)
    else:
        launch(a, 'run_r10_catalog_fifo', options, prefix.with_suffix('.driver.log'))


def prepare(a, spec):
    folder = a.root / spec['directory']
    manifest = folder / 'manifest.json'
    block = spec['block']
    d = json.loads(manifest.read_bytes())
    assert (d['seed'], d['head_tokens'], d.get('head_cached', False)) == (spec['seed'], spec['head_tokens'], spec['head_cached'])
    assert digest(plan(folder, block)[1]) == spec['schedule_sha256']
    catalogue = folder / (block + '-native')
    if not catalogue.with_suffix('.seal.json').exists() or a.dry_run:
        launch(a, 'run_r10_catalog_prepare', dict(block=block, manifest=manifest, plans=folder, probe=a.probe,
                                               model=a.model, runtime=a.runtime, catalogue=catalogue,
                                               device_name=a.device, device_index=a.device_index, timeout_s=240),
               catalogue.with_suffix('.driver.log'))
    return folder, block, catalogue


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('design', 'root', 'worker', 'mechanism-worker', 'probe', 'runtime', 'model', 'observer', 'cost-profile', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--section', choices=('thinking', 'resources', 'mechanism'), action='append', required=True)
    ap.add_argument('--device', default='Vulkan0')
    ap.add_argument('--device-index', type=int, default=1)
    ap.add_argument('--max-new-steps', type=int, help='stop after this many new commands; prepared catalogues and completed arms do not count')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if a.max_new_steps is not None and a.max_new_steps < 0:
        ap.error('--max-new-steps must be nonnegative')
    a.started = 0
    assert len(set(a.section)) == len(a.section)
    design = json.loads(a.design.read_bytes())
    assert design['scope'] == 'R10 calibration only; no primary holdout input or timing'
    execution = dict(design_sha256=digest(a.design), device=a.device, device_index=a.device_index,
                     sha256={k: digest(getattr(a, k)) for k in ('worker', 'mechanism_worker', 'probe', 'observer', 'cost_profile')})
    if not a.dry_run:
        a.output.mkdir(parents=True, exist_ok=True)
        marker = a.output / 'design.json'
        if not marker.exists():
            write_new(marker, execution)
        elif json.loads(marker.read_bytes()) != execution:
            raise ValueError('method controls are already bound to another design or binary')
    for section in a.section:
        if section in ('thinking', 'resources'):
            register = design['thinking_retry' if section == 'thinking' else 'resource']
            name = register['block']
            plans, catalogue = a.root / 'native-plans', a.root / 'native-catalogs' / register['catalogue']
            if 'schedule_sha256' in register:
                assert digest(plan(plans, name)[1]) == register['schedule_sha256']
            setups = [(f'thinking062-{label}', '82.25', {}) for label in ('first', 'second')] if section == 'thinking' else [
                ('resource-' + cap.replace('.', ''), cap, dict(resources=True, monitor_wddm=True)) for cap in register['capacity_order']]
            for prefix, capacity, extras in setups:
                timed(a, a.output / prefix, a.worker, plans, name, catalogue, capacity, register['arrival_scale'], extras)
        else:
            inputs = {}
            for pair in design['mechanism_pairs']:
                key = pair['input']
                if key not in inputs:
                    inputs[key] = prepare(a, design['mechanism_inputs'][key])
                folder, block, catalogue = inputs[key]
                for arm in pair['order']:
                    extras = dict(policy='forced', forced_actions=folder / f'{block}-{arm}.json', publish_rest_on=pair['anchor'],
                                  ssd_hold_ms=pair['hold_ms'], timeout_s=150, chunk_events=True, pipeline_events=True)
                    timed(a, a.output / f'{pair["tag"]}-{arm}', a.mechanism_worker, folder, block, catalogue, pair['capacity'], 1, extras)


if __name__ == '__main__':
    try:
        main()
    except QueueBoundary as e:
        print(f'STOP {e}', flush=True)
