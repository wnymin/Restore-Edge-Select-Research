"""Continue the registered service study after the base calibration and probe exit."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from scripts.validation.check_r10_native_catalog import digest


OUT = Path('outputs/r10')
EVIDENCE = Path('docs/evidence')
DESIGN = Path('configs/r10-service-calibration-v1.json')
CALIBRATION = OUT / 'service-calibration-v1'
HOLDOUT = OUT / 'service-holdout-v1'
CATALOGUES = OUT / 'service-native-catalogs-v1'
PLANS = OUT / 'native-plans'
EXTRA_PLANS = OUT / 'service-native-plans'
MODEL = Path('data/raw/prefill/qwen35-4b-q4km.gguf')
RUNTIME = OUT / 'llama-native-vulkan-build/bin'
WORKER = OUT / 'r10_catalog_worker_service_sse_v1.exe'
OBSERVER = OUT / 'vulkan-observer-pipeline/r10_vk_trace_layer.dll'
HOLDOUT_COST = Path('configs/r10-local-holdout-cost-v1.json')


def read(path):
    return json.loads(path.read_bytes())


def call(module, **flags):
    command = [sys.executable, '-B', '-m', f'scripts.gpu.{module}']
    for name, values in flags.items():
        for val in values if isinstance(values, list) else [values]:
            command += ['--' + name.replace('_', '-'), str(val)]
    subprocess.run(command, check=True)


def stopped():
    for folder in (CALIBRATION, HOLDOUT):
        if (folder / 'stop.request').is_file():
            print(f'STOP {folder / "stop.request"}', flush=True)
            return True
    return False


def evidence(stem):
    return EVIDENCE / (stem + '-summary.json'), EVIDENCE / (stem + '.zip')


def verified(summary, archive, extensions, base_count):
    info = read(summary)
    if (info['archive_sha256'] != digest(archive) or info['design_sha256'] != digest(DESIGN) or
            info['extensions_sha256'] != [digest(p) for p in extensions] or
            len(info['arms']) != base_count + sum(len(read(p)['points']) for p in extensions)):
        raise ValueError(f'calibration summary does not reconstruct the registered ladder: {summary}')
    return info


def calibrate():
    design = read(DESIGN)
    strata = len(design['block_order']) * len(design['profile_order'])
    base_count = strata * len(design['rates_order'])
    max_steps = strata * design['extension_rule']['max_added_rates_per_stratum']
    summary, archive = evidence('r10-native-service-calibration-v1')
    extensions = []
    for i in range(1, max_steps + 2):
        info = verified(summary, archive, extensions, base_count)
        if stopped():
            return None
        if info['all_anchors_found']:
            return summary, archive
        if i > max_steps:
            raise RuntimeError('calibration continuation exceeded the registered added-rate budget')
        plan = EVIDENCE / f'r10-native-service-calibration-extension-{i:02}.json'
        if not plan.is_file():
            call('experiments.plan_r10_service_calibration_extension', design=DESIGN,
                 calibration_summary=summary, calibration_archive=archive, out=plan)
        if not plan.is_file():
            print(f'STOP unfilled anchors after registered calibration: {info["chosen"]}', flush=True)
            return None
        added = read(plan)
        if (added['baseline_summary_sha256'], added['baseline_archive_sha256']) != (digest(summary), digest(archive)):
            raise ValueError(f'extension was not derived from the preceding audited calibration: {plan}')
        extensions.append(plan)
        summary, archive = evidence(f'r10-native-service-calibration-extended-{i:02}')
        if summary.exists() != archive.exists():
            raise FileExistsError(f'partial evidence retained: {summary}, {archive}')
        if not summary.exists():
            n = len(added['points'])
            print(f'Calibration extension {i}: {n} new arms.', flush=True)
            call('experiments.run_r10_service_calibration', design=DESIGN, root=OUT, worker=WORKER, runtime=RUNTIME,
                 model=MODEL, observer=OBSERVER, cost_profile='configs/r10-pilot-cost-v1.json', output=CALIBRATION,
                 max_new_steps=n, extension=extensions)
            if stopped():
                return None
            call('analysis.summarize_r10_service_calibration', design=DESIGN, plans=PLANS, runs=CALIBRATION,
                 archive=archive, out=summary, extension=extensions)
    raise AssertionError('unreachable calibration loop boundary')


def run_holdout(lock, membership, count):
    call('experiments.run_r10_service_holdout', lock=lock, membership=membership, plans=PLANS,
         extra_plans=EXTRA_PLANS, probe=OUT / 'r10_prepare_catalog_target.exe', worker=WORKER, model=MODEL,
         runtime=RUNTIME, observer=OBSERVER, cost_profile=HOLDOUT_COST, catalogues=CATALOGUES,
         out=HOLDOUT, max_new_pairs=count)


def finish_holdout(lock, membership, cases):
    first = cases[0]['id']
    pair = HOLDOUT / first / 'pair.json'
    if not pair.is_file():
        raise FileNotFoundError(f'first service pair has not closed: {pair}')
    audit = EVIDENCE / 'r10-native-service-first-pair-audit.json'
    if not audit.is_file():
        call('analysis.audit_r10_service_first_pair', lock=lock, plans=PLANS, extra_plans=EXTRA_PLANS,
             campaign=HOLDOUT, catalogues=CATALOGUES, out=audit)
    checked = read(audit)
    if (checked['case'], checked['service_lock_sha256'], checked['pair_sha256'], checked['passed']) != (
            first, digest(lock), digest(pair), True):
        raise ValueError('the first service pair must pass the correctness gate before the remaining pairs')
    if stopped():
        return
    remaining = sum(not (HOLDOUT / row['id'] / 'pair.json').is_file() for row in cases)
    if remaining:
        print(f'First service pair verified. Starting the remaining {remaining} registered pairs.', flush=True)
        run_holdout(lock, membership, remaining)
    if stopped():
        return
    summary, archive = evidence('r10-native-service-holdout')
    if summary.exists() != archive.exists():
        raise FileExistsError(f'partial service evidence retained: {summary}, {archive}')
    if not summary.exists():
        call('analysis.summarize_r10_service_holdout', lock=lock, plans=PLANS, extra_plans=EXTRA_PLANS,
             campaign=HOLDOUT, archive=archive, out=summary)
    result = read(summary)
    if (result['lock_sha256'], result['archive_sha256'], [p['id'] for p in result['pairs']]) != (
            digest(lock), digest(archive), [c['id'] for c in cases]):
        raise ValueError('formal service summary differs from the frozen order or archive')
    delivery = EVIDENCE / 'r10-native-service-holdout-delivery-audit.json'
    if not delivery.is_file():
        call('analysis.audit_r10_service_delivery', lock=lock, summary=summary, archive=archive, first_gate=audit,
             campaign=HOLDOUT, catalogues=CATALOGUES, membership=membership, plans=PLANS, extra_plans=EXTRA_PLANS, out=delivery)
    identity = read(delivery)
    if (identity['lock_sha256'], identity['summary_sha256'], identity['archive_sha256'], identity['first_gate_sha256'],
            identity['completed_pairs']) != (digest(lock), digest(summary), result['archive_sha256'], digest(audit), len(cases)):
        raise ValueError('service delivery audit refers to different evidence')
    curves = EVIDENCE / 'r10-native-service-holdout-curves.svg'
    if not curves.is_file():
        call('analysis.plot_r10_service_curves', summary=summary, out=curves)
    print(f'Service holdout summary, delivery audit and curves saved: {summary}, {delivery}, {curves}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--first-pair', action='store_true', help='start only the first service pair after freezing')
    mode.add_argument('--finish-holdout', action='store_true', help='audit the closed first pair, then finish and archive the rest')
    a = parser.parse_args()
    if stopped():
        return
    probe_summary, probe_archive = evidence('r10-native-two-sequence-v1')
    probe = read(probe_summary)
    if probe['archive_sha256'] != digest(probe_archive):
        raise ValueError('two-sequence probe archive differs from its terminal receipt')
    print(f'Two-sequence probe completed (passed={probe["passed"]}); calibrating single-slot service.', flush=True)
    calibration = calibrate()
    if calibration is None or stopped():
        return
    summary, archive = calibration
    membership = EVIDENCE / 'r10-local-service-holdout-layout.json'
    lock = EVIDENCE / 'r10-local-service-holdout-execution-lock.json'
    if not lock.is_file():
        call('experiments.freeze_r10_service_holdout', membership=membership,
             primary_lock=EVIDENCE / 'r10-local-holdout-execution-lock.json',
             primary_summary=EVIDENCE / 'r10-native-primary-holdout-summary.json', design=DESIGN,
             calibration_summary=summary, calibration_archive=archive, calibration_stamp=CALIBRATION / 'design.json',
             worker=WORKER, cost_profile=HOLDOUT_COST, plans=PLANS, extra_plans=EXTRA_PLANS, out=lock)
    registered = read(lock)
    if registered['calibration_summary_sha256'] != digest(summary):
        raise ValueError('holdout lock refers to another calibration summary')
    if a.first_pair and not stopped():
        first = registered['cases'][0]['id']
        if not (HOLDOUT / first / 'pair.json').is_file():
            run_holdout(lock, membership, 1)
        print(f'First service pair boundary: {first}. Audit it before starting the remaining pairs.', flush=True)
    elif a.finish_holdout and not stopped():
        finish_holdout(lock, membership, registered['cases'])


if __name__ == '__main__':
    main()
