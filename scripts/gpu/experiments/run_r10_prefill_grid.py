"""Run native prefill or paired overlap calibration with raw output and device receipts."""

import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from time import monotonic

from scripts.gpu.experiments.run_r10_catalog_prepare import ERROR_MODE, VK_OPTIONS
from scripts.validation.check_r10_native_catalog import digest


def gpu_environment():
    fields = ('name,driver_version,temperature.gpu,clocks.current.sm,clocks.current.memory,'
              'memory.used,memory.total,power.draw,pstate')
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=' + fields, '--format=csv'],
                                capture_output=True, text=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
        return dict(return_code=result.returncode, data=result.stdout, error=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(error=str(exc))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'runtime', 'probe', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--repeats', type=int, default=12)
    ap.add_argument('--seed', type=int, default=10371)
    ap.add_argument('--device-index', type=int, default=1)
    ap.add_argument('--device', default='Vulkan0')
    ap.add_argument('--timeout-s', type=int, required=True)
    ap.add_argument('--observer', type=Path, help='joint-grid Vulkan witness DLL')
    ap.add_argument('--transfer-objects', type=Path, help='joint-grid directory containing 0.bin–3.bin')
    a = ap.parse_args()
    if not 1 <= a.repeats <= 50 or a.device_index < 0 or a.timeout_s <= 0:
        ap.error('invalid repeats, device index, or timeout')
    joint = bool(a.observer and a.transfer_objects)
    if bool(a.observer) != bool(a.transfer_objects):
        ap.error('joint grid needs both --observer and --transfer-objects')
    raw, errors, receipt = [a.out.with_suffix(x) for x in ('.csv', '.stderr', '.run.json')]
    if any(p.exists() for p in (raw, errors, receipt)):
        raise FileExistsError('select a new output prefix')
    runtime = a.runtime.resolve()
    env = os.environ.copy()
    env.pop('VK_ADD_LAYER_PATH', None)
    env.pop('VK_INSTANCE_LAYERS', None)
    env.update(VK_OPTIONS)
    env['GGML_VK_VISIBLE_DEVICES'] = str(a.device_index)
    env['PATH'] = str(runtime) + os.pathsep + env.get('PATH', '')
    report = dict(start_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'), seed=a.seed, repeats=a.repeats,
                  model_sha256=digest(a.model), probe_sha256=digest(a.probe), vulkan_switches=VK_OPTIONS,
                  dll_sha256={p.name:digest(p) for p in sorted(runtime.glob('*.dll'))}, device=a.device,
                  device_index=a.device_index, telemetry_before=gpu_environment())
    argv = [str(a.probe.resolve()), str(a.model.resolve()), str(runtime), a.device]
    report['mode'] = 'joint' if joint else 'isolated'
    if joint:
        observer = a.observer.resolve()
        env.update(VK_ADD_LAYER_PATH=str(observer.parent), VK_INSTANCE_LAYERS='VK_LAYER_R10_timeline_observe')
        payloads = {f'{i}.bin': digest(a.transfer_objects / f'{i}.bin') for i in range(4)}
        report.update(observer_sha256=digest(observer), payloads=payloads)
        argv.extend((str(a.transfer_objects.resolve()), str(observer)))
    argv.extend((str(a.repeats), str(a.seed)))
    started = monotonic()
    previous = ctypes.windll.kernel32.SetErrorMode(ERROR_MODE)
    try:
        with raw.open('xb') as stdout, errors.open('xb') as stderr:
            proc = subprocess.run(argv, env=env, stdout=stdout, stderr=stderr, timeout=a.timeout_s, creationflags=subprocess.CREATE_NO_WINDOW)
        report['return_code'] = proc.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        report['error'] = str(exc)
        raise
    finally:
        ctypes.windll.kernel32.SetErrorMode(previous)
        report['duration_s'] = monotonic() - started
        report['telemetry_after'] = gpu_environment()
        with receipt.open('x', encoding='utf8', newline='\n') as f:
            print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), file=f)
    if proc.returncode:
        raise RuntimeError(f'Cost probe failed: 0x{proc.returncode & 0xffffffff:08x}; see {errors}')
    print(f'Completed {report["mode"]} cost grid; raw log at {raw}')


if __name__ == '__main__':
    main()
