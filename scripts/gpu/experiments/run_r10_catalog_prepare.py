"""Prepare and seal one untimed native catalogue in a guarded subprocess."""

import argparse
from datetime import datetime, timezone
import ctypes
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.gpu.analysis.check_r10_native_artifacts import seal
from scripts.gpu.experiments import r10_holdout_lock
from scripts.validation.check_r10_native_catalog import digest


VK_OPTIONS = {
    'GGML_VK_ASYNC_USE_TRANSFER_QUEUE': '1',
    'GGML_VK_R10_CONCURRENT_BUFFERS': '1',
}
ERROR_MODE = 0x0001 | 0x0002 | 0x8000  # suppress GUI error boxes for the launched probe


def paths(directory):
    parent, stem = directory.parent, directory.name
    return {key: parent / (stem + ext) for key, ext in {
        'csv': '.csv', 'stderr': '.stderr', 'run': '.run.json', 'seal': '.seal.json',
    }.items()}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def write_new(path, payload):
    with path.open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), file=f)


def launch(a):
    if os.name != 'nt':
        raise OSError('this probe uses Windows native unbuffered file I/O')
    a.model, a.runtime, a.probe, a.catalogue = (x.resolve() for x in (a.model, a.runtime, a.probe, a.catalogue))
    sidecar = paths(a.catalogue)
    manifest = json.loads(a.manifest.read_bytes())
    found = [b for b in manifest['blocks'] if b['id'] == a.block]
    if len(found) != 1:
        raise ValueError('one declared block is required')
    split = found[0]['split']
    if a.holdout_lock and a.service_lock:
        raise ValueError('provide at most one execution lock')
    if a.service_lock:
        from scripts.gpu.experiments import r10_service_lock

        r10_service_lock.preparation(r10_service_lock.load(a.service_lock), found[0], a)
    elif a.holdout_lock and split in ('reserved', 'holdout'):
        r10_holdout_lock.preparation(r10_holdout_lock.load(a.holdout_lock), found[0], a)
    elif a.holdout_lock or split not in ('pilot', 'calibration'):
        raise ValueError('untimed reserved preparation needs its frozen execution lock; calibration must omit it')
    plan = (a.plans / f'{a.block}.plan').resolve()
    schedule = (a.plans / f'{a.block}.schedule.json').resolve()
    if digest(plan) != found[0]['wire_sha256'] or digest(schedule) != found[0]['schedule_sha256']:
        raise ValueError('input files have changed since source planning')
    if not a.catalogue.parent.is_dir() or a.catalogue.exists() or any(p.exists() for p in sidecar.values()):
        raise FileExistsError('the output and all of its sidecar files must be new')
    env = os.environ.copy()
    env.pop('VK_INSTANCE_LAYERS', None)
    env.pop('VK_ADD_LAYER_PATH', None)
    env.update(VK_OPTIONS)
    env['GGML_VK_VISIBLE_DEVICES'] = str(a.device_index)
    env['PATH'] = str(a.runtime) + os.pathsep + env.get('PATH', '')
    argv = [str(a.probe), str(a.model), str(a.runtime), a.device_name, str(plan), str(a.catalogue)]
    receipt = dict(block=a.block, start_utc=utc_now(), device_index=a.device_index, device_name=a.device_name,
                   probe_sha256=digest(a.probe), model_sha256=digest(a.model), plan_sha256=digest(plan),
                   dll_sha256={p.name: digest(p) for p in sorted(a.runtime.glob('*.dll'))},
                   vulkan_options=VK_OPTIONS, timeout_s=a.timeout_s)
    if a.holdout_lock:
        receipt['holdout_lock_sha256'] = digest(a.holdout_lock)
    if a.service_lock:
        receipt['service_lock_sha256'] = digest(a.service_lock)
    code = None
    start = time.monotonic()
    getmode = ctypes.windll.kernel32.SetErrorMode
    previous = getmode(ERROR_MODE)
    try:
        with sidecar['csv'].open('xb') as out, sidecar['stderr'].open('xb') as err:
            process = subprocess.run(argv, env=env, stdout=out, stderr=err,
                                     creationflags=subprocess.CREATE_NO_WINDOW, timeout=a.timeout_s)
            code = process.returncode
        receipt['status'] = 'exited'
        receipt['return_code'] = code
        receipt['return_code_hex'] = f'0x{code & 0xFFFFFFFF:08x}'
    except subprocess.TimeoutExpired:
        receipt['status'] = 'killed after timeout'
        raise
    except OSError as exc:
        receipt['status'] = 'launch failed'
        receipt['error'] = str(exc)
        raise
    finally:
        getmode(previous)
        receipt.update(finish_utc=utc_now(), duration_s=time.monotonic() - start)
        write_new(sidecar['run'], receipt)
    if code != 0:
        raise RuntimeError(f"native probe exited with {receipt['return_code_hex']}; see {sidecar['stderr']}")
    result = seal(a.manifest, a.block, a.plans, a.catalogue, sidecar['csv'], sidecar['stderr'])
    result['run_receipt_sha256'] = digest(sidecar['run'])
    for key in ('holdout_lock_sha256', 'service_lock_sha256'):
        if key in receipt:
            result[key] = receipt[key]
    write_new(sidecar['seal'], result)
    print(f"Sealed {len(result['checkpoints'])} native checkpoints in {a.block}.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--block', required=True)
    for name in ('manifest', 'plans', 'probe', 'model', 'runtime', 'catalogue'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--device-index', type=int, default=1)
    ap.add_argument('--device-name', default='Vulkan0')
    ap.add_argument('--holdout-lock', type=Path, help='frozen lock required for untimed preparation of a primary holdout block')
    ap.add_argument('--service-lock', type=Path, help='separate frozen lock for untimed preparation of a service holdout block')
    ap.add_argument('--timeout-s', type=int, required=True)
    a = ap.parse_args()
    if a.timeout_s <= 0 or a.device_index < 0:
        ap.error('timeout must be positive, and device index nonnegative')
    launch(a)


if __name__ == '__main__':
    main()
