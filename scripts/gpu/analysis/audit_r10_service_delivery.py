"""Verify the complete service archive and each on-disk native catalogue."""

import argparse
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path
from zipfile import ZipFile

from scripts.gpu.analysis.check_r10_native_artifacts import need, seal
from scripts.gpu.experiments.r10_service_lock import load
from scripts.gpu.experiments.run_r10_catalog_prepare import paths
from scripts.validation.check_r10_native_catalog import digest


def read(path):
    return json.loads(path.read_bytes())


def catalogue_identity(block, lock, lock_hash, membership, plans, directory):
    side = paths(directory)
    prepared = read(side['run'])
    env = lock['environment']
    expected = dict(block=block, status='exited', return_code=0, service_lock_sha256=lock_hash,
                    device_index=env['device_index'], device_name=env['device'], probe_sha256=env['catalogue_probe_sha256'],
                    model_sha256=env['model_sha256'], dll_sha256=env['runtime_sha256'])
    need(all(prepared[k] == v for k, v in expected.items()), f'{block}: altered preparation identity')
    rebuilt = seal(membership, block, plans, directory, side['csv'], side['stderr'])
    need(prepared['plan_sha256'] == rebuilt['wire_sha256'], f'{block}: preparation receipt uses a different input')
    rebuilt.update(run_receipt_sha256=digest(side['run']), service_lock_sha256=lock_hash)
    need(json.loads(json.dumps(rebuilt)) == read(side['seal']), f'{block}: catalogue no longer matches seal')
    return dict(seal_sha256=digest(side['seal']), objects=len(rebuilt['objects']), checkpoints=len(rebuilt['checkpoints']))


def audit(lock_path, summary_path, archive_path, first_gate, campaign, catalogues, membership, plans, extra_plans):
    lock = load(lock_path)
    lock_hash, archive_hash = digest(lock_path), digest(archive_path)
    summary = read(summary_path)
    need((summary['lock_sha256'], summary['archive_sha256']) == (lock_hash, archive_hash), 'archive or lock differs from summary')
    need(digest(membership) == lock['membership_sha256'], 'membership differs from lock')
    native, counts = {}, Counter()
    for row in lock['blocks']:
        block = row['id']
        block_plans = extra_plans if row['family'] == 'oasst' else plans
        for suffix, field in (('.plan', 'wire_sha256'), ('.schedule.json', 'schedule_sha256')):
            need(digest(block_plans / (block + suffix)) == row[field], f'{block}: {suffix} differs from lock')
        native[block] = catalogue_identity(block, lock, lock_hash, membership, block_plans, catalogues / block)

    gate = read(first_gate)
    first = lock['cases'][0]
    first_ledger = campaign / first['id'] / 'pair.json'
    need(gate['passed'] is True and gate['equal_greedy_outputs_and_finish_reasons'] is True and
         (gate['case'], gate['service_lock_sha256'], gate['pair_sha256'], gate['seal_sha256']) ==
         (first['id'], lock_hash, digest(first_ledger), native[first['block']]['seal_sha256']), 'wrong or failed first-pair gate')

    expected = set()
    with ZipFile(archive_path) as z:
        def member(name, expected_hash):
            need(name not in expected, f'duplicate evidence reference: {name}')
            with z.open(name) as f:
                need(file_digest(f, 'sha256').hexdigest() == expected_hash, f'changed archive member: {name}')
            expected.add(name)

        member('execution-lock.json', lock_hash)
        marker = campaign / 'campaign.json'
        need(read(marker)['service_lock_sha256'] == lock_hash, 'wrong campaign marker')
        member('campaign.json', digest(marker))
        for log in sorted(campaign.glob('*.prepare.log')):
            member(log.name, digest(log))
        for case, result in zip(lock['cases'], summary['pairs'], strict=True):
            case_id = case['id']
            pair_path = campaign / case_id / 'pair.json'
            ledger = read(pair_path)
            need(ledger['service_lock_sha256'] == lock_hash and all(case[k] == ledger[k] == result[k] for k in case),
                 f'{case_id}: pair identity or registration changed')
            member(f'{case_id}/pair.json', digest(pair_path))
            for i, (policy, arm) in enumerate(zip(case['arm_order'], ledger['arms'], strict=True), 1):
                label = f'{i}-{policy}'
                prefix = campaign / case_id / label
                receipt = prefix.with_suffix('.arm.json')
                need((i, policy, label) == (arm['order'], arm['policy'], arm['prefix']) and read(receipt) == arm,
                     f'{case_id}/{label}: unexpected arm receipt')
                member(f'{case_id}/{receipt.name}', digest(receipt))
                failed = bool(arm['return_code'])
                need(result['arms'][policy]['failed'] is failed, f'{case_id}/{label}: wrong summary status')
                required = {'.driver.log'} if failed else {'.driver.log', '.stderr', '.run.json', '.native.csv'}
                need(required <= arm['files'].keys(), f'{case_id}/{label}: missing raw logs')
                for suffix, h in arm['files'].items():
                    path = prefix.with_suffix(suffix)
                    need(digest(path) == h, f'{case_id}/{path.name}: changed local raw file')
                    member(f'{case_id}/{path.name}', h)
                if not failed:
                    run = read(prefix.with_suffix('.run.json'))
                    need(run['seal_sha256'] == native[case['block']]['seal_sha256'], f'{case_id}/{label}: incorrect catalogue')
                counts['failed' if failed else 'successful'] += 1
        actual = z.namelist()
        need(len(actual) == len(expected) and set(actual) == expected, 'archive contains duplicate or unregistered files')

    return dict(scope='service evidence identity and archive integrity; excludes performance calculations',
                lock_sha256=lock_hash, summary_sha256=digest(summary_path), archive_sha256=archive_hash,
                first_gate_sha256=digest(first_gate), archive_members=len(expected), completed_pairs=len(lock['cases']),
                successful_arms=counts['successful'], failed_arms=counts['failed'], catalogues=native)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('lock', 'summary', 'archive', 'first-gate', 'campaign', 'catalogues', 'membership', 'plans', 'extra-plans', 'out'):
        parser.add_argument('--' + key, required=True, type=Path)
    a = parser.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    result = audit(a.lock, a.summary, a.archive, a.first_gate, a.campaign, a.catalogues, a.membership, a.plans, a.extra_plans)
    with a.out.open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(result, sort_keys=True, indent=2), file=f)


if __name__ == '__main__':
    main()
