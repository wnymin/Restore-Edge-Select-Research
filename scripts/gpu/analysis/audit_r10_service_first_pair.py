"""Check the first service pair before timing the rest; report correctness only."""

import argparse
import json
from pathlib import Path

from scripts.gpu.analysis.summarize_r10_service_holdout import evaluate, read
from scripts.gpu.experiments.r10_service_lock import check_environment, load
from scripts.validation.check_r10_native_catalog import digest


def audit(lock_path, plans, extra_plans, campaign, catalogues):
    lock = load(lock_path)
    check_environment(lock)
    lock_hash = digest(lock_path)
    if read(campaign / 'campaign.json')['service_lock_sha256'] != lock_hash:
        raise ValueError('first service pair belongs to a different lock')
    case = lock['cases'][0]
    block = next(b for b in lock['blocks'] if b['id'] == case['block'])
    schedule = (extra_plans if block['family'] == 'oasst' else plans) / (block['id'] + '.schedule.json')
    if digest(schedule) != block['schedule_sha256']:
        raise ValueError('first service pair schedule differs from the lock')
    seal_path = catalogues / (block['id'] + '.seal.json')
    seal = read(seal_path)
    if (seal['block'], seal['service_lock_sha256']) != (block['id'], lock_hash):
        raise ValueError('first service catalogue seal differs from the lock')
    seal_hash = digest(seal_path)
    root = campaign / case['id']
    pair_path = root / 'pair.json'
    ledger = read(pair_path)
    if ledger['service_lock_sha256'] != lock_hash or any(ledger[k] != v for k, v in case.items()):
        raise ValueError('first service pair receipt differs from the lock')
    correctness, outputs = {}, {}
    for i, (policy, arm) in enumerate(zip(case['arm_order'], ledger['arms'], strict=True), 1):
        label = f'{i}-{policy}'
        prefix = root / label
        if (arm['order'], arm['policy'], arm['prefix']) != (i, policy, label) or read(prefix.with_suffix('.arm.json')) != arm:
            raise ValueError(f'invalid arm identity: {prefix}')
        required = {'.driver.log'} if arm['return_code'] else {'.driver.log', '.stderr', '.run.json', '.native.csv'}
        if not required <= arm['files'].keys():
            raise ValueError(f'missing arm evidence: {prefix}')
        for ext, expected in arm['files'].items():
            if digest(prefix.with_suffix(ext)) != expected:
                raise ValueError(f'changed arm evidence: {prefix}{ext}')
        if not arm['return_code'] and read(prefix.with_suffix('.run.json'))['seal_sha256'] != seal_hash:
            raise ValueError(f'wrong native catalogue used: {prefix}')
        row = evaluate(prefix, schedule, arm, block, case, lock, lock_hash)
        correctness[policy] = dict(return_code=arm['return_code'])
        if not row['failed']:
            outputs[policy] = row['token_output']
            correctness[policy].update(completed=row['client']['completed'],
                                       visible=row['client']['ttft_visible_ms']['samples'],
                                       emitted_tokens=sum(len(r['tokens']) for r in row['token_output']))
    equal = len(outputs) == 2 and outputs['local-cost'] == outputs['entry-replay']
    return dict(scope='first registered service pair transport and output gate; excludes performance metrics',
                case=case['id'], service_lock_sha256=lock_hash, pair_sha256=digest(pair_path), seal_sha256=seal_hash,
                requests_per_arm=block['requests'], arms=correctness, equal_greedy_outputs_and_finish_reasons=equal,
                passed=equal and all(a['completed'] == block['requests'] and a['visible'] > 0 for a in correctness.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('lock', 'plans', 'extra-plans', 'campaign', 'catalogues', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    data = audit(a.lock, a.plans, a.extra_plans, a.campaign, a.catalogues)
    with a.out.open('x', encoding='utf8') as f:
        print(json.dumps(data, sort_keys=True, indent=2), file=f)
    if not data['passed']:
        p.exit(1, 'first service pair did not pass; recorded evidence and raw files are preserved\n')


if __name__ == '__main__':
    main()
