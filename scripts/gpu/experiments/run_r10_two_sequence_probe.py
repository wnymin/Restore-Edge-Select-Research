"""Run and preserve the non-timed two-sequence probe after the primary campaign."""

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.gpu.analysis.audit_r10_two_sequence_probe import comparison_rows, mixed_microbatches
from scripts.gpu.experiments.r10_holdout_lock import load, model_backend
from scripts.gpu.experiments.run_r10_catalog_prepare import ERROR_MODE, VK_OPTIONS
from scripts.validation.check_r10_native_catalog import digest


def primary_complete(lock, lock_hash, archive_hash, summary):
    layout = [(row['block'], row['round'], row['arm_order']) for row in lock['run_pairs']]
    rows = [(row['block'], row['round'], row['arm_order']) for row in summary['pair_rows']]
    if summary['lock_sha256'] != lock_hash or layout != rows or summary.get('archive_sha256') != archive_hash:
        raise ValueError('two-sequence probe requires the complete primary summary and archived primary evidence')


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        print(json.dumps(value, indent=2, sort_keys=True), file=f)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('primary-lock', 'primary-summary', 'primary-archive', 'probe', 'model', 'runtime', 'out', 'archive', 'summary'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--timeout-s', default=180, type=int)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if not 0 < a.timeout_s <= 600 or any(path.exists() for path in (a.out, a.archive, a.summary)):
        p.error('select new output paths and a timeout in 1..600 seconds')
    if not (a.probe.is_file() and a.model.is_file() and a.runtime.is_dir()):
        p.error('compiled probe, model or runtime path unavailable')
    lock = load(a.primary_lock)
    lock_hash = digest(a.primary_lock)
    fixed = lock['environment']
    gpu_env = dict(VK_OPTIONS, GGML_VK_VISIBLE_DEVICES=str(fixed['device_index']), LLAMA_BATCH_DEBUG='1')
    cmd = [str(args.resolve()) for args in (a.probe, a.model, a.runtime)] + [fixed['device']]
    if a.dry_run:
        print(json.dumps(dict(command=cmd, environment=gpu_env, would_require_complete_primary=True, timeout_s=a.timeout_s)))
        return
    primary_archive_hash = digest(a.primary_archive)
    primary_complete(lock, lock_hash, primary_archive_hash, json.loads(a.primary_summary.read_bytes()))
    model_backend(lock, a.model, a.runtime, fixed['device'], fixed['device_index'])
    a.out.mkdir()
    stdout, stderr, manifest = (a.out / name for name in ('positions.csv', 'microbatches.log', 'run.json'))
    env = dict(os.environ, **gpu_env)
    env['PATH'] = str(a.runtime.resolve()) + os.pathsep + env['PATH']
    info = dict(scope='synchronous whole-checkpoint two-sequence qualification; no performance timing',
                primary_lock_sha256=lock_hash, primary_summary_sha256=digest(a.primary_summary),
                primary_archive_sha256=primary_archive_hash,
                worker_sha256=digest(a.probe), model_sha256=fixed['model_sha256'], runtime_sha256=fixed['runtime_sha256'],
                command=cmd, environment=gpu_env, timeout_s=a.timeout_s, start_utc=utc())
    errmode = ctypes.windll.kernel32.SetErrorMode
    prior = errmode(ERROR_MODE)
    try:
        with stdout.open('xb') as raw, stderr.open('xb') as debug:
            try:
                rc = subprocess.run(cmd, env=env, stdout=raw, stderr=debug, check=False,
                                    creationflags=subprocess.CREATE_NO_WINDOW, timeout=a.timeout_s).returncode
                info['return_code'] = rc
            except (OSError, subprocess.TimeoutExpired) as e:
                info.update(return_code=None, launch_error=type(e).__name__, detail=str(e))
    finally:
        errmode(prior)
    info.update(finish_utc=utc(), stdout_sha256=digest(stdout), stderr_sha256=digest(stderr))
    findings = None
    if info['return_code'] == 0:
        try:
            with stdout.open(encoding='utf-8-sig', newline='') as f:
                comparisons = comparison_rows(csv.DictReader(f))
            batches = mixed_microbatches(stderr.read_text(encoding='utf-8', errors='replace').splitlines())
            findings = dict(comparisons=comparisons, mixed_submissions=batches)
        except (ValueError, KeyError) as e:
            info['audit_error'] = str(e)
    write_new(manifest, info)
    with ZipFile(a.archive, 'x', compression=ZIP_DEFLATED) as z:
        for path in (stdout, stderr, manifest):
            z.write(path, path.name)
    write_new(a.summary, dict(scope=info['scope'], passed=findings is not None, run=info,
                              archive_sha256=digest(a.archive), findings=findings))
    if findings is None:
        p.exit(1, 'two-sequence qualification did not pass; raw logs and result were preserved\n')


if __name__ == '__main__':
    main()
