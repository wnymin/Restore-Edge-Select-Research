"""Validate and summarize the paired GPU delay control."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import re
from statistics import mean, median


def engine_events(log, total):
    lengths = {total} if isinstance(total, int) else set(total)
    prompt = re.compile(r' id\s+([01]) \| task (\d+) \| new prompt, .*task.n_tokens = (\d+)$')
    cache = re.compile(r' id\s+([01]) \| task (\d+) \| cached n_tokens = (\d+),')
    release = re.compile(r' id\s+([01]) \| task (\d+) \| stop processing:')
    pending, batches, releases, positions = {}, [], {}, defaultdict(list)

    def timestamp(line):
        m, s, ms, us = map(int, line.split()[0].split('.'))
        return 60_000 * m + 1000 * s + ms + us / 1000

    with log.open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            match = prompt.search(line)
            if match and int(match[3]) in lengths:
                pending[int(match[1]), int(match[2])] = (None, int(match[3]))
            match = cache.search(line)
            if match:
                key = (int(match[1]), int(match[2]))
                positions[key].append((timestamp(line), int(match[3])))
                if key in pending and pending[key][0] is None:
                    pending[key] = (int(match[3]), pending[key][1])
            if pending and 'D srv        decode: n_batch (effective)' in line:
                ts = timestamp(line)
                batches.extend((slot, cached, ts, task, length)
                               for (slot, task), (cached, length) in pending.items())
                pending.clear()
            match = release.search(line)
            if match:
                releases[int(match[1]), int(match[2])] = timestamp(line)
    if pending:
        raise ValueError('Full-length prompt has no first batch')
    return batches, releases, positions


def enrich(rows, log, include_peer_position=False):
    totals = {int(r[k]) for r in rows for k in ('total_tokens', 'peer_total_tokens') if int(r[k])}
    batches, releases, positions = engine_events(log, totals)
    expected = sum(1 + bool(int(row['inflight'])) for row in rows)
    if len(batches) != expected:
        raise ValueError(f'Expected {expected} engine batch events, got {len(batches)}')
    events = iter(batches)
    for row in rows:
        # The first capture used the lag as the scheduled-peer flag.
        if row['inflight'] == row.get('lag_ms') and int(row['inflight']) > 0:
            row['inflight'] = '1'
        peer = next(events) if int(row['inflight']) else None
        target = next(events)
        if (target[:2] != (1, int(row['boundary'])) or target[4] != int(row['total_tokens']) or
                (peer and (peer[:2] != (0, 512) or target[2] < peer[2]
                           or peer[4] != int(row['peer_total_tokens'])))):
            raise ValueError(f'Engine events do not match trial: {peer}, {target}')
        row['first_batch_gap_ms'] = f'{target[2] - peer[2]:.3f}' if peer else ''
        row['peer_release_from_target_batch_ms'] = f'{releases[0, peer[3]] - target[2]:.3f}' if peer else ''
        if include_peer_position:
            row['peer_prefix_at_target_batch'] = ''
            if peer:
                key = (0, peer[3])
                row['peer_prefix_at_target_batch'] = (peer[4] if releases[key] <= target[2] else
                    next(n for ts, n in reversed(positions[key]) if ts <= target[2]))


def values(row):
    target, peer, lead = (float(row[k]) for k in ('target_ms', 'peer_ms', 'actual_peer_lead_ms'))
    return target, peer, target + peer, max(peer, lead + target)


def summarize(rows, include_mismatch=False):
    pairs = defaultdict(dict)
    for row in rows:
        key = (int(row['lag_ms']), int(row['seed']), int(row['rep']))
        arm = row['storage']
        if arm in pairs[key] or int(row['inflight']) != 1:
            raise ValueError('Repeated arm or no scheduled peer')
        pairs[key][arm] = row
    by_lag = defaultdict(list)
    for (lag, _, _), block in sorted(pairs.items()):
        if set(block) != {'resident', 'delayed', 'fresh'}:
            raise ValueError('Incomplete paired block')
        resident, delayed, deep = (block[k] for k in ('resident', 'delayed', 'fresh'))
        token_match = len({r['greedy_token'] for r in block.values()}) == 1
        if int(resident['boundary']) != int(delayed['boundary']) or int(deep['boundary']) <= int(resident['boundary']):
            raise ValueError('Unexpected shallow/deep boundary')
        size = int(deep['restored_bytes'])
        if (not 0 < size == int(deep['warm_file_bytes']) <= int(deep['volume_read_bytes'])
                or int(resident['restored_bytes']) or int(delayed['restored_bytes'])):
            raise ValueError(f'Cold SSD read or resident control could not be verified: {deep}')
        for row in block.values():
            row['target_token_match'] = int(token_match)
        if not token_match:
            print(f'Token mismatch: lag={lag}, seed={deep["seed"]}, rep={deep["rep"]}; '
                  f'resident/delayed/deep = {[r["greedy_token"] for r in (resident, delayed, deep)]}')
            if not include_mismatch:
                continue
        by_lag[lag].append((resident, delayed, deep))
    for lag, blocks in sorted(by_lag.items()):
        for i, arm in ((1, 'delayed'), (2, 'fresh')):
            gains = [tuple(a-b for a, b in zip(values(block[0]), values(block[i]), strict=True)) for block in blocks]
            gaps = [float(b[i]['first_batch_gap_ms']) for b in blocks]
            active = sum(float(b[i]['peer_release_from_target_batch_ms']) > 0 for b in blocks)
            print(f'lag={lag} arm={arm} n={len(blocks)} first_batch_gap_median={median(gaps):.2f} '
                  f'peer_active_at_target_batch={active}; '
                  + ' '.join(f'{k}={mean(v):+.2f} ({sum(x > 0 for x in v)}/{len(v)})'
                             for k, v in zip(('target', 'peer', 'sum', 'finish'), zip(*gains), strict=True)))
        baseline = [float(b[0]['first_batch_gap_ms']) for b in blocks]
        baseline_active = sum(float(b[0]['peer_release_from_target_batch_ms']) > 0 for b in blocks)
        mismatch = [float(b[2]['first_batch_gap_ms']) - float(b[1]['first_batch_gap_ms']) for b in blocks]
        print(f'  Resident batch gap median {median(baseline):.2f} ms, peer active {baseline_active}; deep minus delayed: '
              f'signed median {median(mismatch):+.2f} ms, absolute median {median(map(abs, mismatch)):.2f} ms')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', nargs='+', type=Path)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--include-mismatch', action='store_true', help='include divergent tokens in timing sensitivity')
    args = parser.parse_args()
    if args.log and len(args.csv) != 1:
        parser.error('Log association needs exactly one raw CSV')
    rows = []
    for path in args.csv:
        with path.open(encoding='utf-8', newline='') as stream:
            rows.extend(csv.DictReader(stream))
    if not rows:
        parser.error('No trials found')
    if args.log:
        enrich(rows, args.log)
    rows = [r for r in rows if int(r['rep']) >= 0]
    summarize(rows, args.include_mismatch)
    if args.out:
        with args.out.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
