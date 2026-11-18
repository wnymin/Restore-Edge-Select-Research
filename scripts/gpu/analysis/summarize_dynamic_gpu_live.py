"""Check and summarize decisions executed on the live GPU request path."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from scripts.gpu.analysis.summarize_dynamic_gpu_delay import enrich, values


def verify(rows):
    blocks = defaultdict(dict)
    for r in rows:
        key = tuple(int(r[k]) for k in ('seed', 'rep', 'peer_total_tokens', 'lag_ms'))
        policy = r['policy']
        pending = int(r['peer_pending_at_selection'])
        peer = int(r['inflight'])
        if int(r['initial_boundary']) != 1024 or policy in blocks[key] or policy not in {'pending', 'deep', 'shallow'}:
            raise ValueError('Trial did not start at the common boundary or has an invalid arm')
        if peer != int(key[2] > 0) or pending not in (0, 1) or pending > peer:
            raise ValueError('Invalid selection snapshot')
        deep = policy == 'deep' or (policy == 'pending' and pending)
        expected = ('fresh', 1152) if deep else ('resident', 1024)
        if (r['storage'], int(r['boundary'])) != expected:
            raise ValueError('Executed decision disagrees with frozen rule')
        read = int(r['restored_bytes'])
        if deep and not 0 < read == int(r['warm_file_bytes']) <= int(r['volume_read_bytes']):
            raise ValueError('Cold restore was not verified')
        if not deep and read:
            raise ValueError('Shallow reuse restored a file')
        if not 0 <= int(r['decision_ns']) / 1e6 <= float(r['submit_offset_ms']) <= float(r['target_ms']):
            raise ValueError('Decision timing is inconsistent')
        blocks[key][policy] = r
    for key, block in blocks.items():
        if set(block) != {'pending', 'deep', 'shallow'}:
            raise ValueError(f'Incomplete policy block {key}')
        match = len({r['greedy_token'] for r in block.values()}) == 1
        for r in block.values():
            r['target_token_match'] = int(match)
    return blocks


def response(row):
    return values({**row, 'actual_peer_lead_ms': row['actual_peer_lead_ms'] or '0'})


def summarize(blocks):
    print(f'{len(blocks)} blocks, all tokens match in '
          f'{sum(int(block["shallow"]["target_token_match"]) for block in blocks.values())}, '
          f'pending flag agrees in {sum(len({r["peer_pending_at_selection"] for r in b.values()}) == 1 for b in blocks.values())}')
    conditions = defaultdict(list)
    for key, block in blocks.items():
        conditions[key[2:]].append(block)
    for cond, cases in [(('all',), list(blocks.values())), *sorted(conditions.items())]:
        for policy in ('deep', 'pending'):
            gains = [tuple(a-b for a, b in zip(response(block['shallow']), response(block[policy]), strict=True))
                     for block in cases]
            ns = [int(b[policy]['decision_ns']) for b in cases]
            deep = sum(int(b[policy]['boundary']) == 1152 for b in cases)
            metrics = tuple(round(mean(v), 2) for v in zip(*gains))
            print(f'{cond} {policy}: deep={deep}/{len(cases)} target/peer/sum/finish gains={metrics}; '
                  f'sum wins={sum(g[2] > 0 for g in gains)}, worst target loss={max(0, *(-g[0] for g in gains)):.2f}; '
                  f'decision_us median={median(ns)/1000:.3f}, max={max(ns)/1000:.3f}')
    for i, label in enumerate(('target', 'peer', 'sum', 'finish')):
        values_by_block = [tuple(response(b[k])[i] for k in ('pending', 'shallow', 'deep')) for b in blocks.values()]
        regret = mean(dyn - min(shallow, deep) for dyn, shallow, deep in values_by_block)
        gain = mean(deep - dyn for dyn, _, deep in values_by_block)
        print(f'{label}: dynamic gain vs fixed deep = {gain:+.2f} ms; '
              f'dynamic minus fixed-plan paired minimum = {regret:+.2f} ms')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    with args.csv.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if args.log:
        enrich(rows, args.log, include_peer_position=True)
    rows = [r for r in rows if int(r['rep']) >= 0]
    if not rows:
        parser.error('No measured trials')
    blocks = verify(rows)
    summarize(blocks)
    match = {key: b for key, b in blocks.items() if int(b['shallow']['target_token_match'])}
    if match and len(match) != len(blocks):
        print('Sensitivity: blocks with the same target token only')
        summarize(match)
    if args.out:
        with args.out.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
