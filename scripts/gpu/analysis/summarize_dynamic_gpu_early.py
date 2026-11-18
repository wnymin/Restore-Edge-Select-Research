"""Verify release-aligned paired GPU restores and report wall-clock gains."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import mean, median

from scripts.gpu.analysis.summarize_dynamic_gpu_delay import enrich


def metrics(row):
    target_end = float(row['target_finish_ms'])
    target = target_end - float(row['release_ms'])
    peer = float(row['peer_finish_ms'])
    return target, peer, target + peer, max(target_end, peer)


def verify(rows):
    blocks = defaultdict(dict)
    for r in rows:
        key = tuple(int(r[k]) for k in ('seed', 'rep', 'peer_total_tokens', 'release_ms'))
        lead = int(r['requested_lead_ms'])
        release, now, submit, end = (float(r[k]) for k in ('release_ms', 'release_observed_ms', 'target_submit_ms', 'target_finish_ms'))
        if lead in blocks[key] or not release <= now <= submit <= end or int(r['inflight']) != 1:
            raise ValueError('Invalid selection or target release')
        size = int(r['file_bytes'])
        if lead >= 0:
            start, restored = float(r['restore_start_ms']), float(r['restore_end_ms'])
            if not (release - lead <= start <= restored <= submit and int(r['boundary']) == 1152
                    and 0 < size == int(r['restored_bytes']) <= int(r['volume_read_bytes'])):
                raise ValueError(f'Invalid deep restore {r}')
            r['restore_wait_after_release_ms'] = f'{max(0., restored-release):.6f}'
            r['pre_release_ready_ms'] = f'{max(0., release-restored):.6f}'
        elif lead == -1 and int(r['boundary']) == 1024 and not size and not int(r['restored_bytes']):
            r['restore_wait_after_release_ms'] = ''
            r['pre_release_ready_ms'] = ''
        else:
            raise ValueError(f'Invalid shallow reuse {r}')
        blocks[key][lead] = r
    leads = set(next(iter(blocks.values())))
    if not {-1, 0} <= leads or any(set(b) != leads for b in blocks.values()):
        raise ValueError('Missing paired controls')
    for block in blocks.values():
        match = len({r['greedy_token'] for r in block.values()}) == 1
        for r in block.values():
            r['target_token_match'] = int(match)
    return blocks, sorted(leads)


def summarize(blocks, leads):
    print(f'Blocks: {len(blocks)}; token agrees: {sum(int(b[-1]["target_token_match"]) for b in blocks.values())}')
    all_rows = [r for b in blocks.values() for r in b.values()]
    print(f'Max target release-clock overshoot: {max(float(r["release_observed_ms"])-float(r["release_ms"]) for r in all_rows):.2f} ms; '
          f'target finishes last in {sum(float(r["target_finish_ms"]) >= float(r["peer_finish_ms"]) for r in all_rows)}/{len(all_rows)}')
    groups = defaultdict(list)
    for key, b in blocks.items():
        groups[key[2]].append(b)
    for case, group in [('all', list(blocks.values())), *sorted(groups.items())]:
        for lead in leads:
            rows = [b[lead] for b in group]
            if rows[0].get('first_batch_gap_ms'):
                first = median(float(r['first_batch_gap_ms']) for r in rows)
                print(f'{case} lead={lead}: engine first-batch gap median={first:.2f} ms')
            if lead >= 0:
                wait = median(float(r['restore_wait_after_release_ms']) for r in rows)
                ready = sum(float(r['pre_release_ready_ms']) > 0 for r in rows)
                host = median(float(r['restore_end_ms']) - float(r['restore_start_ms']) for r in rows)
                server = median(float(r['restore_server_ms']) for r in rows)
                hold = max(float(r['pre_release_ready_ms']) for r in rows)
                print(f'{case} lead={lead}: restore host/server medians {host:.2f}/{server:.2f} ms; '
                      f'ready={ready}/{len(rows)}, max pre-release hold={hold:.2f} ms, median residual wait={wait:.2f} ms')
            for base in (-1, 0):
                if base == lead:
                    continue
                gains = [tuple(a-b for a, b in zip(metrics(r[base]), metrics(r[lead]), strict=True)) for r in group]
                joint = sum(g[0] > 0 and g[3] > 0 for g in gains)
                print(f'  {case} lead={lead} vs={base}: target/peer/sum/finish gains '
                      f'{tuple(round(mean(g[i] for g in gains), 2) for i in range(4))} ms; '
                      f'joint target+finish improvement {joint}/{len(rows)}, '
                      f'worst peer loss {max(0, *(-g[1] for g in gains)):.2f} ms')


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
    blocks, leads = verify(rows)
    summarize(blocks, leads)
    matched = {k: b for k, b in blocks.items() if int(b[-1]['target_token_match'])}
    if matched and len(matched) < len(blocks):
        print('Sensitivity: blocks with the same target token only')
        summarize(matched, leads)
    if args.out:
        with args.out.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
