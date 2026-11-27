"""Build, run and archive the independent two-sequence staged-restore qualification."""

import argparse
import csv
import ctypes
from hashlib import file_digest
import json
import os
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.gpu.analysis.audit_r10_two_sequence_probe import mixed_microbatches
from scripts.gpu.experiments.r10_holdout_lock import load, model_backend
from scripts.gpu.experiments.run_r10_catalog_prepare import ERROR_MODE, VK_OPTIONS, paths, utc_now, write_new
from scripts.validation.check_r10_native_catalog import digest


OUT = Path('outputs/r10')
EVIDENCE = Path('docs/evidence')
SCOPE = 'two-sequence shared device staging qualification; fixed token history and batch order; not a service A/B'
PAGE = 65536


def read(path):
    return json.loads(path.read_bytes())


def verify_source(wire, schedule, catalogue, fixed):
    hashes = {'wire_sha256': digest(wire), 'schedule_sha256': digest(schedule)}
    side = paths(catalogue)
    seal, prepared = read(side['seal']), read(side['run'])
    if ((seal['block'], seal['split'], prepared['block'], prepared['status'], prepared['return_code']) !=
            ('oasst-cal-03', 'calibration', 'oasst-cal-03', 'exited', 0) or
            any(seal[key] != h for key, h in hashes.items()) or prepared['plan_sha256'] != hashes['wire_sha256'] or
            digest(side['run']) != seal['run_receipt_sha256'] or
            (prepared['model_sha256'], prepared['dll_sha256'], prepared['device_name'], prepared['device_index']) != (
                fixed['model_sha256'], fixed['runtime_sha256'], fixed['device'], fixed['device_index'])):
        raise ValueError('not the sealed calibration fixture')
    req, deps = next((i, c['load_order']) for i, row in enumerate(read(schedule)['trace']['requests'])
                     for c in row['candidates'] if c['boundary_tokens'] == 512)
    if len(deps) != 2:
        raise ValueError('expected one KV and one recurrent object')
    checkpoint = catalogue / f'{deps[-1]}.meta'
    check = seal['checkpoints'][str(deps[-1])]
    if digest(checkpoint) != check['metadata_sha256'] or check['dependencies'] != [int(i) for i in deps]:
        raise ValueError('calibration restore metadata differs from its seal')
    sizes, objects = [], {}
    for ident in deps:
        item = catalogue / f'{ident}.bin'
        obj = seal['objects'][str(ident)]
        if item.stat().st_size != obj['bytes'] or digest(item) != obj['sha256']:
            raise ValueError(f'fixture object {ident} differs from its seal')
        sizes.append(obj['bytes'])
        objects[str(ident)] = obj['sha256']
    return dict(hashes, request=req, object_ids=list(map(int, deps)), bytes=sizes,
                checkpoint_meta_sha256=check['metadata_sha256'], object_sha256=objects, seal_sha256=digest(side['seal']),
                preparation_receipt_sha256=seal['run_receipt_sha256'], preparation_probe_sha256=prepared['probe_sha256'])


def audit(csv_file, log_file, fixture):
    ids, sizes = fixture['object_ids'], fixture['bytes']
    cap = sum(((n + PAGE - 1) // PAGE) * PAGE for n in sizes)
    with csv_file.open(newline='', encoding='ascii') as f:
        rows = [(r[0], list(map(int, r[1:]))) for r in csv.reader(f)]
    if len(rows) != 21 or rows[0] != ('arena', [cap, *ids]):
        raise ValueError('missing actual arena or staged comparison rows')
    for order in range(2):
        reject, join, ram, issue, ssd, a, occupied, b, isolated, compare = rows[1 + 10*order:11 + 10*order]
        for row, ident, size, disk in zip((ram, ssd), ids, sizes, (0, sizes[1]), strict=True):
            key, (got, ticket, n, disk_n) = row
            if (key, got, n, disk_n) != ('landed', ident, size, disk) or ticket <= 0:
                raise ValueError('missing or invalid object upload and device completion ticket')
        if not ram[1][1] < ssd[1][1] or (reject, join, issue, a, occupied, b, isolated, compare) != (
                ('capacity-reject', [cap - PAGE, 0, 1]), ('join-flying', [ids[0], ram[1][1], cap, 2, 2, 0, 0]),
                ('issued', [ids[1], ssd[1][1], cap, 0, 0]), ('copied', [0, sum(sizes), cap]),
                ('occupied-reject', [0, 0, 0, 1]), ('copied', [1, sum(sizes), 0]), ('isolated', [1, 0, 0, 0]),
                ('comparison', [order, 1, 1, 1, 1, 1, 1, 9])):
            raise ValueError('shared pages, two device copies or exact comparison failed')
    sub = mixed_microbatches(log_file.read_text(encoding='utf8', errors='replace').splitlines())
    for order in (False, True):
        shape = [r['microbatches'] for r in sub if r['b_prefill_first'] == order]
        if shape[0] != shape[1]:
            raise ValueError('staged and recomputed references did not use the same physical microbatches')
    if not any(r['restored_b'] and r['any_microbatch_contains_both_sequences'] for r in sub):
        raise ValueError('staged sequence never shared a device microbatch with its live peer')
    return dict(arena_bytes=cap, aggregate_uploaded_payload_per_order=sum(sizes), ssd_bytes_per_order=sizes[1],
                exact_logit_positions=36, exact_checkpoint_states=4, exact_final_states=4,
                prefix_lease_consumers=2, one_page_short_rejections=2, in_flight_joins=2, occupied_sequence_rejections=2,
                zero_charge_after_final_release=2, mixed_submissions=sub)


def invoke(args, stdout, stderr, timeout, env=None):
    with stdout.open('xb') as a, stderr.open('xb') as b:
        return subprocess.run(args, stdout=a, stderr=b, env=env, timeout=timeout,
                              check=False, creationflags=subprocess.CREATE_NO_WINDOW).returncode


def stopped():
    return any((OUT / name / 'stop.request').is_file() for name in ('service-calibration-v1', 'service-holdout-v1'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT / 'two-sequence-staged-v1')
    parser.add_argument('--archive', type=Path, default=EVIDENCE / 'r10-native-two-sequence-staged-v1.zip')
    parser.add_argument('--summary', type=Path, default=EVIDENCE / 'r10-native-two-sequence-staged-v1-summary.json')
    parser.add_argument('--optimization', choices=('O1', 'O0'), default='O1')
    parser.add_argument('--shared-kv-stream', action='store_true', help='use the same KV stream count as the single-slot checkpoint')
    parser.add_argument('--sealed-reference', action='store_true', help='compare sequence B to synchronous restore of the same sealed checkpoint')
    args = parser.parse_args()
    if args.sealed_reference and not args.shared_kv_stream:
        parser.error('--sealed-reference requires --shared-kv-stream')
    if os.name != 'nt' or any(p.exists() for p in (args.out, args.archive, args.summary)):
        parser.error('requires Windows and new output paths')
    if stopped():
        print('Paused: staged probe not started.')
        return
    lock = load(EVIDENCE / 'r10-local-holdout-execution-lock.json')
    fixed = lock['environment']
    build = OUT / 'llama-native-vulkan-build'
    runtime = build / 'bin'
    source_tree = OUT / 'upstream/llama.cpp-8a56aedd6'
    model = Path('data/raw/prefill/qwen35-4b-q4km.gguf')
    model_backend(lock, model, runtime, fixed['device'], fixed['device_index'])
    plan = OUT / 'native-plans/oasst-cal-03.plan'
    catalogue = OUT / 'native-catalogs/oasst-cal-03-overlapped-v1'
    fixture = verify_source(plan, plan.with_suffix('.schedule.json'), catalogue, fixed)
    source = Path(__file__).with_name('r10_two_sequence_staged_probe.cpp')
    args.out.mkdir()
    exe = args.out / 'r10_two_sequence_staged_probe.exe'
    compile_args = ['g++.exe', '-std=c++17', '-' + args.optimization, '-fno-forward-propagate', '-fno-reorder-blocks',
                    '-Wall', '-Wextra', '-I' + str(source_tree / 'include'), '-I' + str(source_tree / 'ggml/include'), str(source),
                    '-L' + str(build / 'src'), '-L' + str(build / 'ggml/src'), '-lllama', '-lggml', '-lggml-base', '-o', str(exe)]
    gpu = dict(VK_OPTIONS, GGML_VK_VISIBLE_DEVICES=str(fixed['device_index']), LLAMA_BATCH_DEBUG='1')
    env = dict(os.environ, **gpu)
    env.pop('VK_INSTANCE_LAYERS', None)
    env.pop('VK_ADD_LAYER_PATH', None)
    env['PATH'] = str(runtime.resolve()) + os.pathsep + env['PATH']
    run_args = [str(p.resolve()) for p in (exe, model, runtime)] + [fixed['device'], str(plan.resolve()), str(catalogue.resolve())]
    if args.shared_kv_stream:
        run_args.append('--shared-kv-stream')
    if args.sealed_reference:
        run_args.append('--sealed-reference')
    provenance = dict(build_command=compile_args, command=run_args, shared_kv_stream=args.shared_kv_stream,
                      reference='sealed-synchronous-B' if args.sealed_reference else 'recomputed-A-and-B',
                      input=fixture, model_sha256=fixed['model_sha256'],
                      runtime_sha256=fixed['runtime_sha256'], environment=gpu, start_utc=utc_now(),
                      sources={p.name: digest(p) for p in [source, Path(__file__), *[source.with_name(name) for name in
                          ('r10_catalog_transfer.h', 'r10_catalog_source.h', 'r10_catalog_target.h', 'r10_catalog.h', 'r10_stage_pool.h')]]})
    build_out, build_err, csv_file, log_file = (args.out / name for name in ('build.stdout', 'build.stderr', 'events.csv', 'native.log'))
    findings = None
    errmode = ctypes.windll.kernel32.SetErrorMode
    previous = errmode(ERROR_MODE)
    try:
        provenance['build_return_code'] = invoke(compile_args, build_out, build_err, 120)
        if provenance['build_return_code'] == 0 and not stopped():
            provenance['binary_sha256'] = digest(exe)
            provenance['run_return_code'] = invoke(run_args, csv_file, log_file, 180, env=env)
            if provenance['run_return_code'] == 0:
                findings = audit(csv_file, log_file, fixture)
        elif stopped():
            provenance['paused_before_native_launch'] = True
    except (OSError, ValueError, subprocess.TimeoutExpired) as err:
        provenance['error'] = f'{type(err).__name__}: {err}'
    finally:
        errmode(previous)
    files = [p for p in (build_out, build_err, csv_file, log_file) if p.is_file()]
    provenance.update(finish_utc=utc_now(), files={p.name: digest(p) for p in files})
    receipt = args.out / 'run.json'
    write_new(receipt, provenance)
    with ZipFile(args.archive, 'x', compression=ZIP_DEFLATED) as archive:
        for path in (*files, receipt):
            archive.write(path, path.name)
    with ZipFile(args.archive) as archive:
        for name, expected in dict(provenance['files'], **{receipt.name: digest(receipt)}).items():
            with archive.open(name) as member:
                if file_digest(member, 'sha256').hexdigest() != expected:
                    raise ValueError(f'staged qualification archive member changed: {name}')
    write_new(args.summary, dict(scope=SCOPE, passed=findings is not None, run=provenance, findings=findings,
                                archive_sha256=digest(args.archive)))
    print(f'Staged two-sequence result: {args.summary}', flush=True)
    if findings is None:
        parser.exit(1, 'Staged two-sequence qualification did not pass; raw evidence preserved.\n')


if __name__ == '__main__':
    main()
