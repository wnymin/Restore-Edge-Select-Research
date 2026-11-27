"""Verify the completed prefix of the local R10 campaign without inspecting outcomes."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
from statistics import median

from scripts.gpu.experiments.r10_holdout_lock import load
from scripts.validation.check_r10_native_catalog import digest


def read(path):
    return json.loads(path.read_bytes())


def duration(receipt):
    start, end = (datetime.fromisoformat(receipt[k]) for k in ('start_utc', 'end_utc'))
    seconds = (end - start).total_seconds()
    if seconds < 0:
        raise ValueError('negative observed arm wall time')
    return seconds


def inspect(lock_path, campaign, catalogues):
    lock = load(lock_path)
    lock_hash = digest(lock_path)
    if read(campaign / 'campaign.json')['lock_sha256'] != lock_hash:
        raise ValueError('campaign and execution lock differ')
    blocks = {r['id']: r for r in lock['blocks']}
    prefix, remaining, current = [], [], None
    by_source, by_family = defaultdict(list), defaultdict(list)
    count, failed = Counter(), Counter()
    for spec in lock['run_pairs']:
        ident = spec['block']
        block = blocks[ident]
        stem = f"r{spec['round']:02d}-{ident}"
        root = campaign / stem
        marker = root / 'pair.json'
        if not marker.exists():
            if root.exists():
                if remaining:
                    raise ValueError(f'out-of-order partial pair: {stem}')
                current = dict(pair=stem, closed_arm_ledgers=len(list(root.glob('*.arm.json'))))
            remaining.append(block)
            continue
        if remaining:
            raise ValueError(f'out-of-order complete pair: {stem}')
        rec = read(marker)
        if rec['lock_sha256'] != lock_hash or any(rec[k] != v for k, v in spec.items()):
            raise ValueError(f'{stem}: receipt identity mismatch')
        times = []
        for i, (policy, arm) in enumerate(zip(spec['arm_order'], rec['arms'], strict=True), 1):
            expected = f'{i}-{policy}'
            if (arm['order'], arm['prefix'], arm['policy']) != (i, expected, policy) or read(root / (expected + '.arm.json')) != arm:
                raise ValueError(f'{stem}: arm identity differs from original ledger')
            paths = {'receipt_sha256': '.run.json', 'native_sha256': '.native.csv',
                     'failure_sha256': '.failure.json', 'stderr_sha256': '.stderr', 'driver_sha256': '.driver.log'}
            mandatory = {'driver_sha256'} if arm['return_code'] else {'driver_sha256', 'receipt_sha256', 'native_sha256', 'stderr_sha256'}
            if not mandatory <= set(arm):
                raise ValueError(f'{stem}: arm ledger omits a required record')
            for key, ext in paths.items():
                if key in arm and digest(root / (expected + ext)) != arm[key]:
                    raise ValueError(f'{stem}: {expected}{ext} changed')
            seconds = duration(arm)
            times.append(seconds)
            by_source[block['source']].append(seconds)
            by_family[block['family']].append(seconds)
            count[block['family']] += 1
            failed[block['family']] += bool(arm['return_code'])
        prefix.append(dict(block=ident, round=spec['round'], family=block['family'], arm_wall_s=times))
    done_blocks = sorted({r['block'] for r in prefix})
    known_next, missing = 0, Counter()
    for b in remaining:
        source = b['source']
        if source in by_source:
            known_next += 2 * median(by_source[source])
        else:
            missing[source] += 2
    manifest = catalogues or campaign
    sealed = sorted(b['id'] for b in blocks.values() if catalogues and (manifest / (b['id'] + '.seal.json')).is_file())
    return dict(scope='wall time and provenance only; no latency or policy outcome comparison', lock_sha256=lock_hash,
                registered_pairs=len(lock['run_pairs']), completed_pairs=len(prefix), completed_arms_by_family=dict(count),
                failed_arms_by_family=dict(failed), covered_blocks=done_blocks, completed_prefix=prefix,
                first_not_committed=lock['run_pairs'][len(prefix)] if remaining else None,
                open_ledger=current, stop_after_pair_requested=(campaign / 'stop.request').exists(),
                sealed_catalogues=sealed, arm_wall_s_by_source={s: dict(arms=len(v), median=median(v), minimum=min(v), maximum=max(v))
                                                               for s, v in sorted(by_source.items())},
                arm_wall_s_by_family={f: [min(v), max(v)] for f, v in sorted(by_family.items())},
                pending_observed_source_estimate_s=known_next, pending_arms_without_source_wall_data=dict(missing))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('lock', 'campaign'):
        p.add_argument('--' + key, type=Path, required=True)
    for key in ('catalogues', 'out'):
        p.add_argument('--' + key, type=Path)
    a = p.parse_args()
    evidence = inspect(a.lock, a.campaign, a.catalogues)
    if a.out:
        with a.out.open('x', encoding='utf8') as f:
            print(json.dumps(evidence, sort_keys=True, indent=2), file=f)
    else:
        print(json.dumps(evidence, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
