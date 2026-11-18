"""Validate same-boundary GPU pairs and extract the two competing token scores."""

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median

from scripts.gpu.analysis.summarize_dynamic_gpu_delay import enrich


TOKEN_IDS = (34, 44)
ARMS = {('none', 0), ('resident', 1024), ('resident', 1152), ('fresh', 1152)}


def extract(row):
    scores, = json.loads(row.pop('token_scores'))
    if int(row['greedy_token']) != scores['id']:
        raise ValueError('Sample identity does not match its score entry')
    top = {int(item['id']): (rank, item) for rank, item in enumerate(scores['top_logprobs'], 1)}
    for token in TOKEN_IDS:
        rank, item = top[token]
        row[f'raw_logprob_{token}'] = item['logprob']
        row[f'raw_rank_{token}'] = rank
        row[f'text_{token}'] = item['token']
    if scores['id'] in TOKEN_IDS and scores['logprob'] != row[f'raw_logprob_{scores["id"]}']:
        raise ValueError('Selected token score differs from its raw ranking entry')


def summarize(rows):
    blocks, groups = defaultdict(set), defaultdict(list)
    prompts = {r['prompt_sha256'] for r in rows}
    if len(prompts) != 1:
        raise ValueError('Expected repeated runs of the same target prompt')
    for row in rows:
        arm = row['storage'], int(row['boundary'])
        key = (int(row['rep']), int(row['inflight']))
        if arm in blocks[key]:
            raise ValueError('Duplicate arm in a paired block')
        blocks[key].add(arm)
        groups[key[1], *arm].append(row)
        read = int(row['restored_bytes'])
        if arm[0] == 'fresh':
            if not 0 < read == int(row['warm_file_bytes']) <= int(row['volume_read_bytes']):
                raise ValueError('SSD cold read could not be verified')
        elif read:
            raise ValueError('Resident or recompute arm used a restore')
    if any(arms != ARMS for arms in blocks.values()) or len(blocks) != 2 * len({k[0] for k in blocks}):
        raise ValueError('Incomplete paired blocks')
    for (peer, storage, boundary), group in sorted(groups.items()):
        gains = [float(r['raw_logprob_34']) - float(r['raw_logprob_44']) for r in group]
        winners = [r['greedy_token'] for r in group]
        gaps = [float(r['first_batch_gap_ms']) for r in group if r['first_batch_gap_ms']]
        active = sum(float(r['peer_release_from_target_batch_ms']) > 0
                     for r in group if r['peer_release_from_target_batch_ms'])
        timing = f'; first batch gap median {median(gaps):.2f} ms, peer still in slot {active}/{len(group)}' if gaps else ''
        print(f'peer={peer} {storage}/{boundary}: token={",".join(winners)}; '
              f'logP(34)-logP(44) {min(gains):+.6f}..{max(gains):+.6f}{timing}')
    for peer in (0, 1):
        resident = {int(r['rep']): r for r in groups[peer, 'resident', 1152]}
        cold = {int(r['rep']): r for r in groups[peer, 'fresh', 1152]}
        equal = sum(all(resident[i][f'raw_logprob_{token}'] == cold[i][f'raw_logprob_{token}']
                        for token in TOKEN_IDS) for i in resident)
        print(f'peer={peer} same-boundary pairs with both raw scores equal: {equal}/{len(resident)}')


def summarize_waits(rows):
    blocks = defaultdict(dict)
    if len({r['prompt_sha256'] for r in rows}) != 1:
        raise ValueError('Expected a single repeated target prompt')
    for row in rows:
        wait = int(row['score_wait_ms'])
        arm = row['storage'], wait
        block = blocks[int(row['rep'])]
        if (arm in block or int(row['inflight']) != 1 or int(row['boundary']) != 1152 or
                row['storage'] not in {'fresh', 'resident'} or (row['storage'] == 'fresh' and wait)):
            raise ValueError(f'Unexpected delay experiment arm: {arm}')
        read = int(row['restored_bytes'])
        if row['storage'] == 'fresh' and not 0 < read == int(row['warm_file_bytes']) <= int(row['volume_read_bytes']):
            raise ValueError('Unverified cold SSD read')
        if row['storage'] == 'resident' and read:
            raise ValueError('Resident arm restored bytes')
        block[arm] = row
    arms = set(next(iter(blocks.values())))
    if not {('fresh', 0), ('resident', 0)} <= arms or any(set(block) != arms for block in blocks.values()):
        raise ValueError('Incomplete wait blocks')
    for arm in sorted(arms):
        group = [block[arm] for block in blocks.values()]
        partitions = defaultdict(list)
        for row in group:
            stage = int(row['peer_prefix_at_target_batch'])
            partitions[stage].append(float(row['raw_logprob_34']) - float(row['raw_logprob_44']))
        gap = median(float(row['first_batch_gap_ms']) for row in group)
        active = sum(float(row['peer_release_from_target_batch_ms']) > 0 for row in group)
        print(f'{arm}: first-batch gap {gap:.2f} ms, peer active {active}/{len(group)}; '
              + '; '.join(f'prefix={k}, n={len(v)}, logP gap={min(v):+.6f}..{max(v):+.6f}'
                          for k, v in sorted(partitions.items())))
        if arm[0] == 'resident':
            matches = [block for block in blocks.values()
                       if block[arm]['peer_prefix_at_target_batch'] == block['fresh', 0]['peer_prefix_at_target_batch']]
            equal = sum(all(block[arm][f'raw_logprob_{token}'] == block['fresh', 0][f'raw_logprob_{token}']
                            for token in TOKEN_IDS) for block in matches)
            same_token = sum(block[arm]['greedy_token'] == block['fresh', 0]['greedy_token'] for block in matches)
            peer_gain = mean(float(block[arm]['peer_ms']) - float(block['fresh', 0]['peer_ms']) for block in blocks.values())
            print(f'  Stage matches vs cold {len(matches)}; both scores equal {equal}; '
                  f'same token {same_token}; cold peer latency gain {peer_gain:+.2f} ms')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    with args.csv.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if args.log:
        enrich(rows, args.log, include_peer_position=bool(rows and 'score_wait_ms' in rows[0]))
    rows = [row for row in rows if int(row['rep']) >= 0]
    if not rows:
        parser.error('No measured rows')
    for row in rows:
        if 'token_scores' in row:
            extract(row)
    if 'score_wait_ms' in rows[0]:
        summarize_waits(rows)
    else:
        summarize(rows)
    if args.out:
        with args.out.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
