"""Lock R10 holdout membership and randomized A/B order using membership metadata only."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from random import Random

from scripts.validation.check_r10_native_catalog import digest


SALT = 'r10-local-holdout-membership-v1'
SEED = 8139702
ROUNDS = 5
ARMS = ('local-cost', 'entry-replay')


def order_key(label):
    return sha256(f'{SALT}:{label}'.encode()).hexdigest()


def plan(native, bailian):
    rows = json.loads(native.read_bytes())['blocks']
    by_id = {r['id']: r for r in rows}
    assert len(by_id) == len(rows)
    sources = defaultdict(list)
    for spec in json.loads(bailian.read_bytes())['cases']:
        if spec['split'] != 'reserved':
            continue
        ident = Path(spec['generated_filename']).stem
        row = by_id[ident]
        assert row['split'] == 'reserved' and row['source_sha256'] == spec['generated_sha256']
        sources[spec['source']].append((ident, spec))
    assert len(sources) == 4 and all(len(group) == 2 for group in sources.values())
    primary, leftover = [], []
    for source in sorted(sources):
        groups = sorted(sources[source], key=lambda x: order_key(x[0]))
        primary.extend(groups[:1])
        leftover.extend(groups[1:])
    leftover.sort(key=lambda x: order_key(x[0]))
    primary.extend(leftover[:2])
    b = [dict(by_id[ident], source=spec['source'], raw_rows=spec['raw_rows'], raw_window_sha256=spec['raw_window_sha256'])
         for ident, spec in sorted(primary)]
    o = [dict(row, source='oasst1') for row in rows if row['family'] == 'oasst' and row['split'] == 'holdout']
    assert len(b) == len(o) == 6 and len({row['schedule_sha256'] for row in b + o}) == 12
    rng = Random(SEED)
    order = {}
    for row in b + o:
        cases = [0, 0, 0, 1, 1] if rng.randrange(2) else [0, 0, 1, 1, 1]
        rng.shuffle(cases)
        order[row['id']] = cases
    runs = []
    for n in range(ROUNDS):
        ids = sorted(order)
        rng.shuffle(ids)
        for ident in ids:
            choices = ARMS[::(-1 if order[ident][n] else 1)]
            runs.append(dict(round=n + 1, block=ident, arm_order=choices))
    return dict(scope='membership and pair ordering exclusively; not a holdout execution authorization',
                source_manifests=dict(native_sha256=digest(native), bailian_sha256=digest(bailian)),
                membership_rule='one hash-ordered window per Bailian source, then two hash-ordered leftovers; all six OASST holdouts',
                membership_salt=SALT, arm_order_seed=SEED, pairs_per_block=ROUNDS, blocks=b + o,
                reserved_not_primary=[ident for ident, _ in leftover[2:]], run_pairs=runs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('native', 'bailian', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    args = ap.parse_args()
    result = plan(args.native, args.bailian)
    with args.out.open('x', encoding='utf8', newline='\n') as out:
        print(json.dumps(result, indent=2, sort_keys=True), file=out)
    print(f"Locked membership of {len(result['blocks'])} blocks and {len(result['run_pairs'])} randomized pairs; no execution authorized.")


if __name__ == '__main__':
    main()
