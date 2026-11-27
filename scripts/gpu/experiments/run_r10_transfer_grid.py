"""Measure the native phase timestamps of a single RAM/SSD object and device scatter."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
from time import monotonic

from scripts.gpu.experiments.run_r10_catalog_prepare import ERROR_MODE, VK_OPTIONS
from scripts.gpu.experiments.run_r10_prefill_grid import gpu_environment
from scripts.validation.check_r10_native_catalog import digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime', 'probe', 'catalogue', 'seal', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--repeats', type=int, default=12)
    ap.add_argument('--seed', type=int, default=10372)
    ap.add_argument('--device-index', type=int, default=1)
    ap.add_argument('--device', default='Vulkan0')
    ap.add_argument('--timeout-s', type=int, default=120)
    a = ap.parse_args()
    if not 1 <= a.repeats <= 50 or a.device_index < 0 or a.timeout_s <= 0:
        ap.error('invalid repeats, device index, or timeout')
    raw, errors, receipt = [a.out.with_suffix(x) for x in ('.csv', '.stderr', '.run.json')]
    if a.out.exists() or any(p.exists() for p in (raw, errors, receipt)):
        raise FileExistsError('select a new output prefix')
    source = json.loads(a.seal.read_bytes())
    if source['split'] not in ('pilot', 'calibration'):
        raise ValueError('only calibration payloads can be used')
    k0, k1 = [int(i) for i, o in source['objects'].items() if o['bytes'] == 16777216][:2]
    ssd = int(next(i for i, o in source['objects'].items() if o['bytes'] == 52690944))
    paths = [a.catalogue / f'{i}.bin' for i in (k0, k1, ssd)]
    for ident, file in zip((k0, k1, ssd), paths, strict=True):
        if digest(file) != source['objects'][str(ident)]['sha256']:
            raise ValueError(f'calibration bytes do not match catalogue seal for {file.name}')
    primary, secondary, checkpoint = (p.read_bytes() for p in paths)
    a.out.mkdir()
    for ident, obj in enumerate((primary[:len(primary) // 2], primary, primary + secondary, checkpoint)):
        (a.out / f'{ident}.bin').write_bytes(obj)
    del primary, secondary, checkpoint
    runtime = a.runtime.resolve()
    env = os.environ.copy()
    env.pop('VK_ADD_LAYER_PATH', None)
    env.pop('VK_INSTANCE_LAYERS', None)
    env.update(VK_OPTIONS)
    env['GGML_VK_VISIBLE_DEVICES'] = str(a.device_index)
    env['PATH'] = str(runtime) + os.pathsep + env.get('PATH', '')
    report = dict(seed=a.seed, repeats=a.repeats, probe_sha256=digest(a.probe), source_seal_sha256=digest(a.seal),
                  payloads={p.name:digest(p) for p in sorted(a.out.glob('*.bin'))},
                  dll_sha256={p.name:digest(p) for p in sorted(runtime.glob('*.dll'))}, device=a.device,
                  device_index=a.device_index, telemetry_before=gpu_environment())
    argv = [str(a.probe.resolve()), str(runtime), a.device, str(a.out.resolve()), str(a.repeats), str(a.seed)]
    started = monotonic()
    previous = ctypes.windll.kernel32.SetErrorMode(ERROR_MODE)
    try:
        with raw.open('xb') as stdout, errors.open('xb') as stderr:
            proc = subprocess.run(argv, env=env, stdout=stdout, stderr=stderr, timeout=a.timeout_s,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
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
        raise RuntimeError(f'Transfer probe failed: 0x{proc.returncode & 0xffffffff:08x}; see {errors}')
    print(f'Completed native transfer grid; raw log at {raw}')


if __name__ == '__main__':
    main()
