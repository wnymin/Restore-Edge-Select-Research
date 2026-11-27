"""Verify R10 independent text blocks from sealed prompts and token manifests."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from scripts.data.build_r10_oasst import CHECKPOINT, LIMIT, digest, shared, token_digest


def audit(manifest, private):
    proof = json.loads(manifest.read_text(encoding='utf-8'))
    assert digest(private) == proof['private_tokens_sha256']
    with gzip.open(private, 'rt', encoding='utf-8') as f:
        payloads = [json.loads(line) for line in f]
    pending = iter(payloads)
    trees, roots, message_ids, prompt_hashes = set(), set(), set(), set()
    counts = Counter()
    result = []
    main_splits = ['calibration'] * 4 + ['holdout'] * 6
    splits = [b['split'] for b in proof['blocks']]
    assert splits in (main_splits, main_splits + ['service-holdout'] * 2)
    assert len({b['id'] for b in proof['blocks']}) == len(splits)
    for block in proof['blocks']:
        firsts = {}
        depths = Counter()
        lengths = []
        coding_trees = 0
        assert len(block['requests']) == 64
        for tick, row in enumerate(block['requests']):
            raw = next(pending)
            assert raw['block'] == block['id'] and raw['request'] == row['request'] and tick == row['tick']
            ids = tuple(raw['tokens'])
            assert len(ids) == row['tokens'] and 128 <= len(ids) <= LIMIT <= proof['context_tokens'] - proof['output_token_limit']
            assert all(type(i) is int and 0 <= i < 250000 for i in ids)
            fingerprint = token_digest(ids)
            assert fingerprint == row['sha256'] and fingerprint not in prompt_hashes
            assert row['source_message'] not in message_ids
            message_ids.add(row['source_message'])
            prompt_hashes.add(fingerprint)
            lengths.append(len(ids))
            key = row['source_tree']
            if key not in firsts:
                assert key not in trees and row['root_hash'] not in roots
                assert row['eligible_boundaries'] == [0] and row['exact_common_prefix'] == 0
                trees.add(key)
                roots.add(row['root_hash'])
                firsts[key] = (ids, row)
            else:
                prev, preceding = firsts.pop(key)
                assert preceding['tick'] == tick - 8 and preceding['coding_tag'] == row['coding_tag']
                assert row['root_hash'] == preceding['root_hash'] and max(row['turns'], preceding['turns']) >= 3
                lcp = shared(prev, ids)
                cut = min(lcp, len(prev) - 1, len(ids) - 1) // CHECKPOINT
                assert lcp == row['exact_common_prefix']
                assert row['eligible_boundaries'] == [CHECKPOINT * n for n in range(cut + 1)]
                coding_trees += row['coding_tag']
                depths[cut * CHECKPOINT] += 1
        assert not firsts and sum(depths.values()) == 32
        reusable = sum(n for depth, n in depths.items() if depth >= CHECKPOINT)
        assert reusable >= 8 and coding_trees >= 4
        counts[block['split']] += len(lengths)
        result.append(dict(block=block['id'], ordinary_and_coding_trees=32, coding_tagged_trees=coding_trees,
                           first_prompts_without_cache=32, paired_prefix_depths=dict(sorted(depths.items())),
                           min_prompt_tokens=min(lengths), max_prompt_tokens=max(lengths)))
    assert next(pending, None) is None and len(trees) == len(roots) == 32 * len(splits)
    assert len(prompt_hashes) == len(message_ids) == 64 * len(splits)
    return dict(scope='actual native model tokens and exact common prefixes, generated arrival ticks; no served latency',
                manifest_sha256=digest(manifest), private_tokens_sha256=digest(private), request_counts=dict(counts),
                unique_session_trees=len(trees), exact_duplicates_across_blocks=0, blocks=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('tokens', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    summary = json.dumps(audit(args.manifest, args.tokens), sort_keys=True, indent=2)
    if args.out:
        with args.out.open('x', encoding='utf-8') as f:
            print(summary, file=f)
    else:
        print(summary)


if __name__ == '__main__':
    main()
