"""Prepare paired shallow/deep synthetic inputs for the native mechanism pilot."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from random import Random

from scripts.data.build_r10_native_plans import write_plan


def blocks(seed=81397, head_tokens=768, head_cached=False):
    rng = Random(seed)

    def prompt(n):
        return [rng.randrange(300, 50000) for _ in range(n)]

    # Keep target/follower draws unchanged when varying the amount of head work.
    first = prompt(768)[:head_tokens]
    base, other, target_tail, follower_tail = prompt(1024), prompt(512), prompt(64), prompt(64)
    for shared in (False, True):
        requests = [dict(id='head', tokens=first, arrival_ms=0, boundaries=[0, 512] if head_cached else [0]),
                    dict(id='target', tokens=base + target_tail, arrival_ms=1, boundaries=[0, 512, 1024]),
                    dict(id='follower', tokens=(base[:512] if shared else other) + follower_tail,
                         arrival_ms=1, boundaries=[0, 512])]
        ident = 'mechanism-shared' if shared else 'mechanism-private'
        yield dict(id=ident, split='calibration', family='synthetic', requests=requests,
                   source_sha256=sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=81397, help='fixed token histories; change only for an independent check')
    ap.add_argument('--head-tokens', type=int, choices=(16, 128, 768), default=768, help='change only preceding work')
    ap.add_argument('--head-cached', action='store_true', help='give the head a private 512-token SSD checkpoint for an in-flight anchor')
    args = ap.parse_args()
    if args.head_cached and args.head_tokens < 512:
        ap.error('a cached predecessor must have at least 512 tokens')
    args.out.mkdir(parents=True, exist_ok=True)
    plans = []
    for block in blocks(args.seed, args.head_tokens, args.head_cached):
        plan = write_plan(block, args.out)
        plans.append(plan)
        for arm, boundary in [('shallow', 512), ('deep', 1024)]:
            path = args.out / (block['id'] + '-' + arm + '.json')
            with path.open('x', encoding='utf8', newline='\n') as f:
                print(json.dumps(dict(schedule_sha256=plan['schedule_sha256'], boundaries=[512 if args.head_cached else 0, boundary, 512]),
                                 sort_keys=True), file=f)
        print(f"Prepared {block['id']} as same-input shallow/deep pair; schedule {plan['schedule_sha256']}")
    with (args.out / 'manifest.json').open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(dict(scope='one calibration seed per family of counterfactuals; not holdout',
                              seed=args.seed, head_tokens=args.head_tokens, head_cached=args.head_cached, blocks=plans),
                         indent=2, sort_keys=True), file=f)


if __name__ == '__main__':
    main()
