"""Run five registered native prefill diagnostic pairs on calibration inputs only."""

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
from subprocess import CREATE_NO_WINDOW, run
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.gpu.analysis.explain_r10_prefill_diagnostic_anchors import peak_in_window
from scripts.gpu.analysis.explain_r10_primary_actions import explain
from scripts.gpu.analysis.summarize_r10_pilot_aa import one
from scripts.gpu.analysis.summarize_r10_pilot_load import measured_window
from scripts.gpu.experiments import r10_holdout_lock as primary
from scripts.gpu.experiments.run_r10_catalog_prepare import write_new
from scripts.gpu.experiments.run_r10_holdout_batch import invoke, snapshot
from scripts.validation.check_r10_native_catalog import digest


OUTPUTS = Path('outputs/r10')
EVIDENCE = Path('docs/evidence')
DESIGN = Path('configs/r10-prefill-diagnostics-v1.json')
COST = Path('configs/r10-pilot-cost-v1.json')
MODEL = Path('data/raw/prefill/qwen35-4b-q4km.gguf')
BUILD = OUTPUTS / 'llama-native-vulkan-build'
RUNTIME = BUILD / 'bin'
UPSTREAM = OUTPUTS / 'upstream/llama.cpp-8a56aedd6'
MIXED = OUTPUTS / 'r10_catalog_worker_warmup_grid.exe'
OBSERVER = OUTPUTS / 'vulkan-observer-pipeline/r10_vk_trace_layer.dll'
SOURCES = Path(__file__).resolve().parent
RUN = 'scripts.gpu.experiments.run_r10_catalog_fifo'
SCOPE = 'R10 GPU-04 native prefill diagnostics on calibration sources only'


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def burst(n, spec):
    group_size, gap, last = (spec[k] for k in ('requests_per_group', 'within_group_ms', 'last_arrival_ms'))
    require(group_size >= 2 and n % group_size == 0 and n >= group_size * 2 and gap > 0, 'invalid burst dimensions')
    width = (group_size - 1) * gap
    period = Fraction(last - width, n // group_size - 1)
    require(period > width, 'burst groups must not overlap or reorder requests')
    return [period * (i // group_size) + gap * (i % group_size) for i in range(n)]


def ram_header(raw):
    # Route both the untimed preload and the timed read through RAM. Preserve
    # the semantic object kind used by restore, prefix identity and KV carry.
    for old, new in ((b"if (obj.kind != 'K') continue;", b'// Diagnostic: preload every object from the same sealed catalogue.'),
                     (b"if (o.kind == 'R') {", b'if (false) { // Diagnostic: all objects use the existing RAM upload path.')):
        require(raw.count(old) == 1, 'unexpected native transfer source; refusing an ambiguous diagnostic build')
        raw = raw.replace(old, new)
    return raw


def store_bytes(path, payload):
    if path.exists():
        require(path.read_bytes() == payload, f'prepared input changed: {path}')
    else:
        with path.open('xb') as out:
            out.write(payload)


def source(spec):
    wire = OUTPUTS / 'native-plans' / (spec['block'] + '.plan')
    schedule = wire.with_suffix('.schedule.json')
    catalogue = OUTPUTS / 'native-catalogs' / spec['catalogue']
    seal = catalogue.with_suffix('.seal.json')
    require([digest(p) for p in (wire, schedule, seal)] == [spec[k] for k in ('wire_sha256', 'schedule_sha256', 'seal_sha256')],
            'registered native calibration inputs changed')
    registered = read(seal)
    require(registered['block'] == spec['block'] and registered['split'] in ('calibration', 'pilot') and
            (registered['wire_sha256'], registered['schedule_sha256']) == (digest(wire), digest(schedule)),
            'only the exact registered calibration or pilot catalogue may be reused')
    return wire, schedule, catalogue, seal


def derive(case, design, parent, dest=None):
    wire, schedule, catalogue, original_seal = parent
    data = read(schedule)
    trace = data['trace']
    rows = trace['requests']
    if case['arrivals'] == 'burst':
        grid = burst(len(rows), design['burst'])
        for r, t in zip(rows, grid, strict=True):
            r['arrival_ms'] = str(t)
    else:
        require(case['arrivals'] == 'original', 'unknown arrival scheme')
        grid = [Fraction(r['arrival_ms']) for r in rows]
    if case['placement'] == 'ram':
        for item in trace['objects']:
            item['tier'] = 'RAM'
    else:
        require(case['placement'] == 'mixed', 'unknown placement')
    geometry = [x['bytes'] for x in trace['objects'] if x['tier'] == 'RAM']
    if dest is not None and (case['arrivals'] != 'original' or case['placement'] != 'mixed'):
        schedule = dest / 'diagnostic.schedule.json'
        seal = dest / 'diagnostic.seal.json'
        # The wire, state artifacts, checkpoint identities and candidates are
        # unchanged. This is a derivative binding, not a new readiness proof.
        store_bytes(schedule, (json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode('utf8'))
        amended = dict(read(original_seal), schedule_sha256=digest(schedule), parent_seal_sha256=digest(original_seal),
                       scope='R10 diagnostic derivative; original wire and state artifacts unchanged',
                       diagnostic_changes={k: case[k] for k in ('arrivals', 'placement')})
        store_bytes(seal, (json.dumps(amended, sort_keys=True, indent=2) + '\n').encode('utf8'))
    else:
        seal = original_seal
    grid = [(x - grid[0]) * Fraction(case['arrival_scale']) for x in grid]
    return wire, schedule, catalogue, seal, grid, sum(geometry)


def paused(out):
    return any((d / 'stop.request').is_file() for d in (out, OUTPUTS / 'service-calibration-v1', OUTPUTS / 'service-holdout-v1'))


def build_ram(out):
    directory = out / 'ram-build'
    directory.mkdir(exist_ok=True)
    source_file = directory / 'r10_catalog_worker.cpp'
    header = directory / 'r10_catalog_transfer.h'
    store_bytes(source_file, (SOURCES / source_file.name).read_bytes())
    store_bytes(header, ram_header((SOURCES / header.name).read_bytes()))
    exe = directory / 'r10_prefill_diagnostic_ram_worker.exe'
    receipt, log = directory / 'build.json', directory / 'build.log'
    if receipt.is_file():
        require(digest(exe) == read(receipt)['binary_sha256'], 'RAM worker changed following build')
        return exe
    require(not exe.exists() and not log.exists(), 'partial diagnostic build preserved')
    args = ['g++.exe', '-std=c++17', '-O1', '-fno-forward-propagate', '-fno-reorder-blocks', '-Wall', '-Wextra',
            '-I' + str(SOURCES), '-I' + str(UPSTREAM / 'include'), '-I' + str(UPSTREAM / 'ggml/include'), str(source_file),
            '-L' + str(BUILD / 'src'), '-L' + str(BUILD / 'ggml/src'), '-lllama', '-lggml', '-lggml-base', '-lpsapi', '-o', str(exe)]
    with log.open('xb') as f:
        run(args, stdout=f, stderr=f, timeout=150, creationflags=CREATE_NO_WINDOW, check=True)
    write_new(receipt, dict(argv=args, worker_source_sha256=digest(source_file), transfer_source_sha256=digest(header),
                            original_transfer_sha256=digest(SOURCES / header.name), binary_sha256=digest(exe)))
    return exe


def execute(a, design, case, parent, fixed):
    dest = a.out / case['id']
    ledger = dest / 'pair.json'
    if ledger.is_file():
        saved = read(ledger)
        require(saved['case'] == case and saved['execution_sha256'] == digest(a.out / 'design.json'), 'previous pair identity changed')
        for key, value in saved['file_sha256'].items():
            require(digest(dest / key) == value, f'archived file changed: {dest / key}')
        print(f'PRESERVE {case["id"]}', flush=True)
        return saved
    worker = build_ram(a.out) if case['placement'] == 'ram' else MIXED
    if paused(a.out):
        return None
    dest.mkdir()
    wire, schedule, catalogue, seal, grid, ram_bytes = derive(case, design, parent, dest)
    data, cost = read(schedule), read(COST)['cost']
    arms, n = [], len(grid)
    print(f'START {case["id"]}', flush=True)
    for idx, policy in enumerate(case['arm_order'], 1):
        prefix = dest / f'{idx}-{policy}'
        options = dict(worker=worker, runtime=RUNTIME, model=MODEL, plan=wire, schedule=schedule, catalogue=catalogue, seal=seal, observer=OBSERVER,
                       out=prefix, policy=policy, device=fixed['device'], device_index=fixed['device_index'], capacity_mib=case['capacity_mib'],
                       arrival_scale=case['arrival_scale'], timeout_s=design['timeout_s'], cost_profile=COST)
        argv = [i for k, v in options.items() for i in ('--' + k.replace('_', '-'), str(v))] + ['--warmup', '--warmup-grid']
        setup = snapshot()
        code = invoke(RUN, argv, prefix.with_suffix('.driver.log'))
        arms.append(dict(policy=policy, return_code=code, setup=setup, after=snapshot(), prefix=prefix.name))

    actions = []
    for arm in arms:
        if arm['return_code']:
            arm.update(audited_completed=0, completion_unknown=True)
            continue
        prefix = dest / arm['prefix']
        recorded, stats, chosen = one(prefix, schedule)
        require((recorded['block'], recorded['policy'], recorded['capacity_bytes'], recorded['resident_ram_bytes'], recorded['seal_sha256'],
                 recorded['cost_profile_sha256'], recorded['worker_sha256'], recorded['observer_sha256']) == (
            read(seal)['block'], arm['policy'], Fraction(case['capacity_mib']) * 2**20, ram_bytes, digest(seal), digest(COST), digest(worker),
            digest(OBSERVER)), 'diagnostic execution identity differs from registered case')
        require(list(map(Fraction, recorded['planned_arrival_offsets_ms'])) == grid and len(chosen) == n,
                'open-loop arrival schedule or completion count changed')
        arrivals, finish = (list(map(Fraction, recorded[k])) for k in ('client_arrivals_ms', 'completions_ms'))
        replay = explain(recorded, data, cost)
        moved = replay['logical_uploaded_bytes_by_tier_and_trigger']
        require(case['placement'] != 'ram' or not any(t.startswith('SSD/') for t in moved), 'unexpected SSD read in the RAM diagnostic')
        arm.update(stats=stats, audited_completed=n, arrivals=measured_window(arrivals, finish),
                   actual_peak_submissions_100ms=peak_in_window(arrivals, 100), **replay)
        actions.append(chosen)
    paired = None
    if len(actions) == 2:
        by = {arm['policy']: arm['stats'] for arm in arms}
        paired = dict(action_disagreements=sum(a != b for a, b in zip(*actions, strict=True)),
                      local_minus_entry_ms={k: by['local-cost'][k] - by['entry-replay'][k]
                                            for k in ('response_mean_ms', 'response_p95_ms')})
    files = {p.name: digest(p) for p in sorted(dest.iterdir()) if p.is_file() and p != schedule}
    result = dict(case=case, requests_per_arm=n, catalogue=read(seal)['block'], schedule_sha256=digest(schedule),
                  wire_sha256=digest(wire), parent_seal_sha256=digest(parent[-1]), execution_sha256=digest(a.out / 'design.json'),
                  planned_peak_submissions_100ms=peak_in_window(grid, 100), planned_last_arrival_ms=float(grid[-1]),
                  file_sha256=files, arms=arms, paired=paired)
    write_new(ledger, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=OUTPUTS / 'prefill-diagnostics-v1')
    p.add_argument('--summary', type=Path, default=EVIDENCE / 'r10-native-prefill-diagnostics-v1-summary.json')
    p.add_argument('--archive', type=Path, default=EVIDENCE / 'r10-native-prefill-diagnostics-v1.zip')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    design = read(DESIGN)
    require(design['scope'] == SCOPE and (design['mixed_worker_sha256'], design['cost_profile_sha256']) == (digest(MIXED), digest(COST)),
            'wrong diagnostic design or native inputs')
    lock = primary.load(EVIDENCE / 'r10-local-holdout-execution-lock.json')
    primary.check_controller(lock)
    sources = {k: source(v) for k, v in design['sources'].items()}
    cases = design['pairs']
    require(len(cases) == len({c['id'] for c in cases}) == 5 and all(sorted(c['arm_order']) == sorted(primary.POLICIES) for c in cases),
            'expected five distinct registered policy pairs')
    timing = {}
    for c in cases:
        _, _, _, _, grid, resident = derive(c, design, sources[c['source']])
        timing[c['id']] = dict(n=len(grid), span_ms=float(grid[-1]), peak_100ms=peak_in_window(grid, 100), ram_bytes=resident)
    spaced, mixed, ram = (timing[f'oasst-{suffix}'] for suffix in ('spaced-mixed', 'burst-mixed', 'burst-ram'))
    require(all(spaced[k] == mixed[k] == ram[k] for k in ('n', 'span_ms')) and
            spaced['peak_100ms'] < mixed['peak_100ms'] == ram['peak_100ms'] and
            spaced['ram_bytes'] == mixed['ram_bytes'] < ram['ram_bytes'], 'registered OASST contrasts differ on extra dimensions')
    if a.dry_run:
        ram_header((SOURCES / 'r10_catalog_transfer.h').read_bytes())
        print(json.dumps(timing, sort_keys=True, indent=2))
        return
    require(not a.summary.exists() and not a.archive.exists(), 'closed or partially archived diagnostics; preserve existing output')
    fixed = lock['environment']
    primary.model_backend(lock, MODEL, RUNTIME, fixed['device'], fixed['device_index'])
    require(digest(OBSERVER) == fixed['observer_sha256'], 'Vulkan timeline observer differs from the frozen binary')
    a.out.mkdir(parents=True, exist_ok=True)
    marker = a.out / 'design.json'
    native_sources = ('r10_catalog_worker.cpp', 'r10_catalog_transfer.h', 'r10_catalog_ledger.h', 'r10_catalog_witness.h',
                      'r10_catalog_source.h', 'r10_catalog_target.h', 'r10_catalog.h', 'r10_stage_pool.h', 'r10_vk_trace_abi.h')
    before = dict(design_sha256=digest(DESIGN), runner_sha256=digest(Path(__file__)), environment=fixed, planned=timing,
                  native_sources_sha256={name: digest(SOURCES / name) for name in native_sources})
    if not marker.exists():
        write_new(marker, before)
    else:
        require(read(marker) == before, 'diagnostic design, runner, or runtime changed after start')
    results = []
    for c in cases:
        if paused(a.out):
            print('STOP before next diagnostic pair', flush=True)
            return
        result = execute(a, design, c, sources[c['source']], fixed)
        if result is None:
            return
        results.append(result)
    with ZipFile(a.archive, 'x', compression=ZIP_DEFLATED) as z:
        z.write(DESIGN, 'registered-design.json')
        z.write(marker, marker.name)
        for p in sorted((a.out / 'ram-build').glob('build.*')):
            z.write(p, p.relative_to(a.out).as_posix())
        for result in results:
            d = a.out / result['case']['id']
            for label in (*result['file_sha256'], 'pair.json'):
                z.write(d / label, (d / label).relative_to(a.out).as_posix())
    with ZipFile(a.archive) as z:
        require(z.testzip() is None, 'diagnostic archive failed CRC verification')
    counts = Counter(arm['return_code'] == 0 for row in results for arm in row['arms'])
    write_new(a.summary, dict(scope=SCOPE, archive_sha256=digest(a.archive), design_sha256=digest(DESIGN),
                              arms_succeeded=counts[True], arms_failed=counts[False], pairs=results))
    print(f'Native prefill diagnostic archive: {a.summary}', flush=True)


if __name__ == '__main__':
    main()
