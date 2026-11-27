"""Run only complete, registered pairs from the separate service holdout lock."""

import argparse
import json
from pathlib import Path

from scripts.gpu.experiments import r10_holdout_lock as primary, r10_service_lock as service
from scripts.gpu.experiments.run_r10_catalog_prepare import paths, utc_now, write_new
from scripts.gpu.experiments.run_r10_holdout_batch import invoke, snapshot
from scripts.validation.check_r10_native_catalog import digest


def read(p):
    return json.loads(p.read_bytes())


def ensure_catalogue(a, block, plans, catalogue):
    sidecars = paths(catalogue)
    if not sidecars['seal'].exists():
        code = invoke('scripts.gpu.experiments.run_r10_catalog_prepare',
                      ['--block', block['id'], '--manifest', a.membership, '--plans', plans, '--probe', a.probe,
                       '--model', a.model, '--runtime', a.runtime, '--catalogue', catalogue, '--service-lock', a.lock,
                       '--device-index', a.device_index, '--device-name', a.device, '--timeout-s', a.prep_timeout_s],
                      a.out / (block['id'] + '.prepare.log'))
        if code:
            raise RuntimeError(f'service catalogue preparation failed for {block["id"]}; preserve the logs')
    seal = read(sidecars['seal'])
    if seal['block'] != block['id'] or seal.get('service_lock_sha256') != digest(a.lock):
        raise ValueError('service catalogue was not prepared using the active service freeze')
    return sidecars['seal']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('lock', 'membership', 'plans', 'extra-plans', 'probe', 'worker', 'model', 'runtime', 'observer',
                'cost-profile', 'catalogues', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--prep-timeout-s', type=int, default=330)
    p.add_argument('--max-new-pairs', type=int, default=1)
    a = p.parse_args()
    if not 1 <= a.max_new_pairs <= 24 or a.prep_timeout_s <= 0:
        p.error('invalid diagnostic batch size or timeout')
    lock = service.load(a.lock)
    service.check_environment(lock)
    env = lock['environment']
    a.device, a.device_index = env['device'], env['device_index']
    primary.model_backend(lock, a.model, a.runtime, a.device, a.device_index)
    if (digest(a.membership), digest(a.worker), digest(a.observer), digest(a.probe), digest(a.cost_profile)) != (
            lock['membership_sha256'], env['worker_sha256'], env['observer_sha256'], env['catalogue_probe_sha256'],
            lock['cost_profile_sha256']):
        raise ValueError('service campaign inputs differ from the lock')
    lock_hash = digest(a.lock)
    a.out.mkdir(parents=True, exist_ok=True)
    a.catalogues.mkdir(parents=True, exist_ok=True)
    mark = a.out / 'campaign.json'
    identity = dict(service_lock_sha256=lock_hash, scope=service.SCOPE)
    if not mark.exists():
        write_new(mark, identity)
    elif read(mark) != identity:
        raise ValueError('output directory belongs to a different service campaign')
    by_id = {b['id']: b for b in lock['blocks']}
    started = 0
    for case in lock['cases']:
        dest = a.out / case['id']
        ledger = dest / 'pair.json'
        if ledger.exists():
            saved = read(ledger)
            if saved['service_lock_sha256'] != lock_hash or any(saved[k] != v for k, v in case.items()):
                raise ValueError('completed service pair does not match its registered order and parameters')
            continue
        if dest.exists():
            raise FileExistsError(f'incomplete pair retained at {dest}; inventory the attempt before retrying')
        if started == a.max_new_pairs or (a.out / 'stop.request').exists():
            break
        block = by_id[case['block']]
        plans = a.extra_plans if block['family'] == 'oasst' else a.plans
        wire, schedule = (plans / (block['id'] + suffix) for suffix in ('.plan', '.schedule.json'))
        primary.block(lock, block['id'], digest(wire), digest(schedule))
        catalogue = a.catalogues / block['id']
        seal = ensure_catalogue(a, block, plans, catalogue)
        dest.mkdir()
        print(f'Service holdout: {dest.name}', flush=True)
        arms = []
        for i, policy in enumerate(case['arm_order'], 1):
            prefix = dest / f'{i}-{policy}'
            before = snapshot()
            start = utc_now()
            code = invoke('scripts.gpu.experiments.run_r10_catalog_fifo',
                          ['--worker', a.worker, '--runtime', a.runtime, '--model', a.model, '--plan', wire, '--schedule', schedule,
                           '--catalogue', catalogue, '--seal', seal, '--observer', a.observer, '--out', prefix, '--policy', policy,
                           '--capacity-mib', lock['capacity_mib'], '--arrival-scale', case['arrival_scale'], '--timeout-s', lock['timeout_s'],
                           '--device', a.device, '--device-index', a.device_index, '--cost-profile', a.cost_profile, '--service-lock',
                           a.lock, '--service-case', case['id'], '--warmup', '--warmup-grid', '--service-sse', '--service-tokens',
                           case['short_tokens'], '--service-long-tokens', case['long_tokens']], prefix.with_suffix('.driver.log'))
            row = dict(policy=policy, order=i, prefix=prefix.name, start_utc=start, end_utc=utc_now(), return_code=code,
                       setup=before, after=snapshot(), files={})
            for suffix in ('.run.json', '.failure.json', '.native.csv', '.stderr', '.driver.log'):
                f = prefix.with_suffix(suffix)
                if f.exists():
                    row['files'][suffix] = digest(f)
            if not code and '.run.json' not in row['files']:
                raise ValueError('missing receipt from successful service arm')
            write_new(prefix.with_suffix('.arm.json'), row)
            arms.append(row)
        write_new(ledger, dict(case, service_lock_sha256=lock_hash, arms=arms))
        started += 1


if __name__ == '__main__':
    main()
