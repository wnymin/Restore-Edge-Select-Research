"""Archive the untimed primary catalogue preparation receipts without reading AB outcomes."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.gpu.analysis.check_r10_native_artifacts import HEADER
from scripts.gpu.experiments.r10_holdout_lock import load
from scripts.gpu.experiments.run_r10_catalog_prepare import paths
from scripts.validation.check_r10_native_catalog import digest


def examine(block, files, lock_hash, environment):
    tag = block['id']
    seal, run = (json.loads(files[k].read_bytes()) for k in ('seal', 'run'))
    if any(seal[k] != block[k] for k in ('family', 'split', 'wire_sha256', 'schedule_sha256')) or \
            seal['block'] != tag or seal['holdout_lock_sha256'] != lock_hash:
        raise ValueError(f'{tag}: catalogue seal does not match frozen block')
    source_fields = {'probe_sha256': 'catalogue_probe_sha256', 'model_sha256': 'model_sha256',
                     'dll_sha256': 'runtime_sha256', 'device_index': 'device_index', 'device_name': 'device'}
    if any(run[k] != environment[v] for k, v in source_fields.items()) or (run['status'], run['return_code'], run['block'],
            run['plan_sha256'], run['holdout_lock_sha256']) != ('exited', 0, tag, block['wire_sha256'], lock_hash):
        raise ValueError(f'{tag}: preparation run does not match frozen execution environment')
    for k, ref in {'run': 'run_receipt_sha256', 'csv': 'producer_csv_sha256', 'stderr': 'producer_stderr_sha256'}.items():
        if digest(files[k]) != seal[ref]:
            raise ValueError(f'{tag}: altered {k} preparation record')
    expected = seal['checkpoints']
    if len(expected) != block['unique_checkpoints'] or len(seal['objects']) != block['unique_checkpoints'] + block['unique_kv_objects']:
        raise ValueError(f'{tag}: incorrect object or checkpoint counts')
    with files['csv'].open(encoding='ascii', newline='') as f:
        raw = csv.DictReader(f)
        if tuple(raw.fieldnames or ()) != HEADER:
            raise ValueError(f'{tag}: unexpected preparation transcript columns')
        rows = [{k: int(v) for k, v in row.items()} for row in raw]
    if [str(r['checkpoint']) for r in rows] != sorted(expected, key=int):
        raise ValueError(f'{tag}: missing or duplicated checkpoint results')
    seen = set()
    for row in rows:
        cp = expected[str(row['checkpoint'])]
        deps = {str(i) for i in cp['dependencies']}
        if not deps <= set(seal['objects']) or (row['exact_state'], row['exact_final_state'], row['exact_logits_positions'],
                cp['exact_logits_positions'], row['depth'], row['dependencies'], row['shared_identity_checks']) != (
                1, 1, 8, 8, cp['depth'], len(deps), len(seen & deps)):
            raise ValueError(f'{tag}: state, continuation or shared identity check did not pass')
        seen.update(deps)
    if seen != set(seal['objects']):
        raise ValueError(f'{tag}: seal contains an unvalidated object')
    return dict(block=tag, family=block['family'], source=block['source'], seal_sha256=digest(files['seal']),
                exact_checkpoints=len(rows), exact_logits_positions=sum(r['exact_logits_positions'] for r in rows),
                checkpoints_by_depth=dict(Counter(r['depth'] for r in rows)), shared_identity_checks=sum(r['shared_identity_checks'] for r in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('lock', 'catalogues', 'archive', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.archive.exists() or a.out.exists():
        raise FileExistsError('select new preparation evidence paths')
    lock, lock_hash = load(a.lock), digest(a.lock)
    files = {b['id']: paths(a.catalogues / b['id']) for b in lock['blocks']}
    rows = [examine(b, files[b['id']], lock_hash, lock['environment']) for b in lock['blocks']]
    with ZipFile(a.archive, 'x', compression=ZIP_DEFLATED) as z:
        z.writestr('execution-lock.json', a.lock.read_bytes())
        for sidecars in files.values():
            for path in sidecars.values():
                z.write(path, path.name)
    with a.out.open('x', encoding='utf-8', newline='\n') as f:
        print(json.dumps(dict(scope='untimed primary preparation sidecars; no AB outcomes or repeated hashing of raw object bytes',
                              lock_sha256=lock_hash, archive_sha256=digest(a.archive), blocks=rows,
                              exact_checkpoints=sum(x['exact_checkpoints'] for x in rows),
                              exact_logits_positions=sum(x['exact_logits_positions'] for x in rows),
                              shared_identity_checks=sum(x['shared_identity_checks'] for x in rows)), indent=2, sort_keys=True), file=f)


if __name__ == '__main__':
    main()
