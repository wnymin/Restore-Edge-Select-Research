"""Run the frozen local holdout in registered order, preserving incomplete pairs."""

import argparse
import ctypes
import json
from pathlib import Path
from subprocess import CREATE_NO_WINDOW, TimeoutExpired, run
import sys

from scripts.gpu.experiments.run_r10_catalog_prepare import paths, utc_now, write_new
from scripts.gpu.experiments import r10_holdout_lock
from scripts.validation.check_r10_native_catalog import digest


class PowerStatus(ctypes.Structure):
    _fields_ = [('ac_online', ctypes.c_ubyte), ('battery_flags', ctypes.c_ubyte), ('battery_percent', ctypes.c_ubyte),
                ('battery_saver', ctypes.c_ubyte), ('remaining_s', ctypes.c_ulong), ('full_s', ctypes.c_ulong)]


def snapshot():
    """Sample before/after an arm; never run telemetry in the timed worker."""
    def command(args):
        try:
            p = run(args, capture_output=True, creationflags=CREATE_NO_WINDOW, timeout=6, check=False)
        except (OSError, TimeoutExpired) as e:
            return dict(error=type(e).__name__)
        return dict(code=p.returncode, output=p.stdout.decode('utf8', errors='replace').splitlines(),
                    errors=p.stderr.decode('utf8', errors='replace').splitlines())

    power = PowerStatus()
    ac = None
    if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power)):
        ac = {0: False, 1: True}.get(power.ac_online)
    return dict(utc=utc_now(), ac_online=ac, power_plan=command(['powercfg', '/GETACTIVESCHEME']), gpu=command([
        'nvidia-smi', '--query-gpu=index,uuid,name,temperature.gpu,clocks.gr,clocks.mem,power.draw,pstate,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits']))


def invoke(module, arguments, log):
    with log.open('xb') as stream:
        completed = run([sys.executable, '-m', module, *map(str, arguments)], stdout=stream, stderr=stream,
                        creationflags=CREATE_NO_WINDOW, check=False)
    return completed.returncode


def prepare(args, row, catalogue):
    expected = paths(catalogue)
    if expected['seal'].exists():
        s = json.loads(expected['seal'].read_bytes())
        if s.get('holdout_lock_sha256') == digest(args.lock) and s['block'] == row['id']:
            return
        raise ValueError('the existing catalogue seal is from another block or execution lock')
    code = invoke('scripts.gpu.experiments.run_r10_catalog_prepare',
                  ['--block', row['id'], '--manifest', args.manifest, '--plans', args.plans, '--probe', args.probe,
                   '--model', args.model, '--runtime', args.runtime, '--catalogue', catalogue, '--holdout-lock', args.lock,
                   '--device-index', args.device_index, '--device-name', args.device, '--timeout-s', args.prep_timeout_s],
                  args.out / (row['id'] + '.prepare.log'))
    if code:
        raise RuntimeError(f'catalogue preparation failed for {row["id"]}; retain logs and stop this campaign')


def timing(args, lock, row, pair, catalogue, root):
    plan = args.plans / (row['id'] + '.plan')
    schedule = args.plans / (row['id'] + '.schedule.json')
    fixed = lock['timing']
    records = []
    for order, policy in enumerate(pair['arm_order'], 1):
        prefix = root / f'{order}-{policy}'
        setup = snapshot()
        start = utc_now()
        code = invoke('scripts.gpu.experiments.run_r10_catalog_fifo',
                      ['--worker', args.worker, '--runtime', args.runtime, '--model', args.model, '--plan', plan, '--schedule', schedule,
                       '--catalogue', catalogue, '--seal', paths(catalogue)['seal'], '--observer', args.observer, '--out', prefix,
                       '--policy', policy, '--capacity-mib', fixed['capacity_mib'], '--arrival-scale', r10_holdout_lock.arrival_scale(lock, row),
                       '--timeout-s', fixed['timeout_s'], '--device', args.device, '--device-index', args.device_index,
                       '--holdout-lock', args.lock, '--round', pair['round'], '--cost-profile', args.cost_profile, '--warmup', '--warmup-grid'],
                      prefix.with_suffix('.driver.log'))
        record = dict(policy=policy, order=order, start_utc=start, end_utc=utc_now(), return_code=code, prefix=prefix.name,
                      setup=setup, after=snapshot())
        for key, suffix in [('receipt_sha256', '.run.json'), ('failure_sha256', '.failure.json'), ('native_sha256', '.native.csv'),
                            ('stderr_sha256', '.stderr'), ('driver_sha256', '.driver.log')]:
            file = prefix.with_suffix(suffix)
            if file.exists():
                record[key] = digest(file)
        if not code and 'receipt_sha256' not in record:
            raise ValueError('successful worker exit is missing its receipt')
        records.append(record)
        # Preserve either arm immediately, including failure or controller timeout.
        write_new(prefix.with_suffix('.arm.json'), record)
    write_new(root / 'pair.json', dict(pair, lock_sha256=digest(args.lock), arms=records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('lock', 'manifest', 'plans', 'probe', 'worker', 'observer', 'model', 'runtime', 'cost-profile', 'catalogues', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--prep-timeout-s', type=int, default=330)
    parser.add_argument('--max-new-pairs', type=int, default=60)
    a = parser.parse_args()
    if not 1 <= a.max_new_pairs <= 60 or a.prep_timeout_s <= 0:
        parser.error('pair limit and preparation timeout must be positive')
    lock = r10_holdout_lock.load(a.lock)
    r10_holdout_lock.check_controller(lock)
    if digest(a.manifest) != lock['native_manifest_sha256']:
        raise ValueError('preparation manifest differs from freeze')
    a.device = lock['environment']['device']
    a.device_index = lock['environment']['device_index']
    r10_holdout_lock.model_backend(lock, a.model, a.runtime, a.device, a.device_index)
    if digest(a.cost_profile) != lock['cost_profile_sha256'] or digest(a.worker) != lock['environment']['worker_sha256'] or \
            digest(a.observer) != lock['environment']['observer_sha256'] or digest(a.probe) != lock['environment']['catalogue_probe_sha256']:
        raise ValueError('check the frozen environment before preparing any input')
    a.out.mkdir(parents=True, exist_ok=True)
    a.catalogues.mkdir(parents=True, exist_ok=True)
    stamp = a.out / 'campaign.json'
    identity = dict(lock_sha256=digest(a.lock), scope='one ordered primary holdout campaign')
    if not stamp.exists():
        write_new(stamp, identity)
    elif json.loads(stamp.read_bytes()) != identity:
        raise ValueError('output directory already belongs to another lock')
    blocks = {r['id']: r for r in lock['blocks']}
    remaining = a.max_new_pairs
    for pair in lock['run_pairs']:
        row = blocks[pair['block']]
        root = a.out / f"r{pair['round']:02d}-{row['id']}"
        if (root / 'pair.json').exists():
            saved = json.loads((root / 'pair.json').read_bytes())
            if saved['lock_sha256'] != identity['lock_sha256'] or any(saved[k] != v for k, v in pair.items()):
                raise ValueError(f'pair is from another lock or order: {root.name}')
            continue
        if root.exists():
            raise FileExistsError(f'incomplete pair retained at {root}; inventory attempts before any rerun')
        if (a.out / 'stop.request').exists():
            print('Stop requested; no new pair started.', flush=True)
            break
        if not remaining:
            break
        catalogue = a.catalogues / row['id']
        prepare(a, row, catalogue)
        root.mkdir()
        print(f"Primary holdout round {pair['round']}, block {row['id']}", flush=True)
        timing(a, lock, row, pair, catalogue, root)
        remaining -= 1


if __name__ == '__main__':
    main()
