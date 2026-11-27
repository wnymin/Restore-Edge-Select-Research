"""Run the archive-only R10 closeout checks after all native workers exit."""

import argparse
from pathlib import Path
from shutil import copyfileobj

from scripts.gpu.analysis.audit_r10_local_deliverables import audit, read
from scripts.gpu.analysis.reproduce_r10_local_results import run
from scripts.validation.check_r10_native_catalog import digest


ROOT = Path('outputs/r10')
EVIDENCE = Path('docs/evidence')
STREAM = 'r10-native-service-holdout'
CSV = 'r10-native-service-requests.csv'
CSV_AUDIT = 'r10-native-service-requests-audit.json'
REPRODUCTION = 'r10-native-service-reproduction-receipt.json'
INDEX = 'r10-local-delivery-audit.json'


def paused():
    return any((ROOT / name / 'stop.request').is_file() for name in (
        'service-calibration-v1', 'service-holdout-v1', 'prefill-diagnostics-v1'))


def publish(source, target):
    if target.exists():
        if digest(source) != digest(target):
            raise ValueError(f'existing evidence differs; preserved {target}')
        return
    with source.open('rb') as src, target.open('xb') as out:
        copyfileobj(src, out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, default=EVIDENCE)
    p.add_argument('--work', required=True, type=Path, help='new path under outputs for reproducible intermediates')
    args = p.parse_args()
    e, work = args.evidence, args.work
    summary, archive, delivery, lock = (e / name for name in (
        STREAM + '-summary.json', STREAM + '.zip', STREAM + '-delivery-audit.json', 'r10-local-service-holdout-execution-lock.json'))
    if paused():
        print('STOP local closeout before reading evidence.', flush=True)
        return
    if not all(f.is_file() for f in (summary, archive, delivery, lock, e / 'r10-native-service-result-explanation.json')):
        p.error('service summary, archive, delivery audit and action explanation must exist')
    work.mkdir()

    if not (e / CSV).is_file() or not (e / CSV_AUDIT).is_file():
        run('export_r10_service_requests', lock=lock, summary=summary, archive=archive, delivery=delivery,
            csv=work / CSV, out=work / CSV_AUDIT)
        publish(work / CSV, e / CSV)
        publish(work / CSV_AUDIT, e / CSV_AUDIT)
    if paused():
        print('STOP local closeout after request export.', flush=True)
        return

    if not (e / REPRODUCTION).is_file():
        path = work / 'service-reproduction'
        run('reproduce_r10_local_results', evidence=e, mode='service', out=path)
        publish(path / 'reproduction-receipt.json', e / REPRODUCTION)
    if paused():
        print('STOP local closeout after service reproduction.', flush=True)
        return

    if (e / INDEX).exists():
        if audit(e) != read(e / INDEX):
            raise ValueError('existing final evidence index differs; original preserved')
        return
    path = work / INDEX
    run('audit_r10_local_deliverables', evidence=e, out=path)
    if missing := read(path)['missing']:
        raise ValueError(f'local closeout incomplete; index left in the working directory: {sorted(missing)}')
    publish(path, e / INDEX)
    print(f'Local evidence index saved: {e / INDEX}', flush=True)


if __name__ == '__main__':
    main()
