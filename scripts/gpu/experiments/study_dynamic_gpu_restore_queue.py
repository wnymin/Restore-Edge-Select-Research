"""Attribute wall time using correlated server task IDs in early-restore trials."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import re
from statistics import median

from scripts.gpu.analysis.summarize_dynamic_gpu_delay import engine_events


STAMP = re.compile(r'^(\d+)\.(\d+)\.(\d+)\.(\d+) ')
PATTERNS = {
    'post': re.compile(r' D que\s+post: new task, id = (\d+)(/\d+)?, front = \d+$'),
    'wait': re.compile(r' D res\s+add_waiting_: add task (\d+) to waiting list\.'),
    'process': re.compile(r' D que\s+process_new_: processing task, id = (\d+)$'),
    'decline': re.compile(r' D srv\s+process_sing: decoding, decline task, id_task = (\d+)$'),
    'defer': re.compile(r' D que\s+defer: defer task, id = (\d+)$'),
    'send': re.compile(r' D res\s+send: sending result for task id = (\d+)$'),
    'launch': re.compile(r' I slot\s+launch_slot_: id\s+1 \| task (\d+) \| processing task,'),
}
IDLE = re.compile(r' [ID] slot process_sing: id\s+(\d+) \| task -1 \| (saving idle slot to prompt cache|__TEST_TAG_CACHE_IDLE_SLOT__)$')
CHECKPOINT = re.compile(r' [ID] slot\s+(?:operator\(\)|create_check): id\s+0 \| task (\d+) \| (main/do_checkpoint = yes,|created context checkpoint )')
RESTORE_METADATA = re.compile(r' D state_read_meta: cell_count = (\d+), dest_seq_id = 1$')
BATCH_CONFIG = re.compile(r' I llama_context: (n_batch|n_ubatch)\s+= (\d+)$')


def read_trace(log):
    events = defaultdict(lambda: defaultdict(list))
    saving, spans, starts = {}, [], []
    creating, checkpoints = {}, defaultdict(list)
    metadata = []
    batch_caps = defaultdict(set)

    def time(line):
        m, s, ms, us = map(int, STAMP.match(line).groups())
        return m * 60000 + s * 1000 + ms + us / 1000

    with log.open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            if cap := BATCH_CONFIG.search(line):
                batch_caps[cap[1]].add(int(cap[2]))
                continue
            if marker := RESTORE_METADATA.search(line):
                metadata.append((time(line), int(marker[1])))
                continue
            checkpoint = CHECKPOINT.search(line)
            if checkpoint:
                task = int(checkpoint[1])
                if checkpoint[2].startswith('main'):
                    creating[task] = time(line)
                elif (start := creating.pop(task, None)) is not None:
                    checkpoints[task].append((start, time(line)))
                continue
            marker = IDLE.search(line)
            if marker:
                slot, tag = int(marker[1]), marker[2]
                if tag.startswith('saving'):
                    saving[slot] = time(line)
                    starts.append(saving[slot])
                else:
                    start = saving.pop(slot, None)
                    if start is None:
                        raise ValueError('Cache-save completion marker without a start')
                    spans.append((start, time(line), slot))
                continue
            for kind, pattern in PATTERNS.items():
                match = pattern.search(line)
                if match:
                    key = 'single' if kind == 'post' and not match[2] else kind
                    events[int(match[1])][key].append(time(line))
                    break
    return events, spans, starts, checkpoints, metadata, batch_caps


def extract(rows, log):
    lengths = {int(r[k]) for r in rows for k in ('total_tokens', 'peer_total_tokens')}
    batches, releases, positions = engine_events(log, lengths)
    events, idle_spans, idle_starts, checkpoints, metadata, batch_caps = read_trace(log)
    if 'batch_tokens' in rows[0]:
        for csv_key, log_key in (('batch_tokens', 'n_batch'), ('ubatch_tokens', 'n_ubatch')):
            if {int(r[csv_key]) for r in rows} != batch_caps[log_key]:
                raise ValueError(f'Actual engine {log_key} differs from the captured configuration')
    if 2 * len(rows) != len(batches):
        raise ValueError('Expected the peer and target first batches for all trials, including warmup')
    single = {k: v for k, v in events.items() if v['wait'] and v['single'] and v['send']}
    for r, peer, target in zip(rows, batches[::2], batches[1::2], strict=True):
        if (peer[:2] + (peer[4],) != (0, 512, int(r['peer_total_tokens'])) or
                target[:2] + (target[4],) != (1, int(r['boundary']), int(r['total_tokens']))):
            raise ValueError('Full prompt task ID association failed')
        target_id = target[3]
        peer_post, = events[peer[3]]['post']
        target_post, = events[target_id]['post']
        launch, = events[target_id]['launch']
        if not peer_post <= target_post <= launch <= target[2]:
            raise ValueError('Invalid completion timeline')
        idle_between = sum(max(0., min(end, target[2])-max(begin, launch)) for begin, end, _ in idle_spans)
        checkpoint_between = sum(end-begin for begin, end in checkpoints[peer[3]] if launch <= begin <= end <= target[2])
        r.update(target_task=target_id, peer_task=peer[3], target_queue_ms=round(launch-target_post, 6),
                 first_batch_gap_ms=round(target[2]-peer[2], 6),
                 target_batch_to_release_ms=round(releases[1, target_id]-target[2], 6),
                 peer_release_from_target_batch_ms=round(releases[0, peer[3]]-target[2], 6),
                 target_idle_save_ms=round(idle_between, 6),
                 target_idle_save_starts=sum(launch <= t < target[2] for t in idle_starts),
                 target_peer_checkpoint_ms=round(checkpoint_between, 6),
                 target_launch_to_batch_ms=round(target[2]-launch, 6), restore_task='', restore_queue_ms='',
                 restore_server_ms_trace='', restore_before_metadata_ms='', restore_metadata_to_result_ms='',
                 restore_dispatch_from_peer_post_ms='', restore_dispatch_from_peer_release_ms='',
                 peer_pos_at_restore_dispatch='', peer_next_pos_after_restore='',
                 peer_next_pos_from_restore_result_ms='',
                 restore_decoding_declines='', restore_slot_defers='',
                 restore_client_outside_server_ms='', restore_result_to_target_queue_ms='')
        candidates = [(tid, e) for tid, e in single.items()
                      if peer_post < e['single'][0] <= e['send'][-1] <= target_post]
        if len(candidates) != int(r['boundary'] == '1152'):
            raise ValueError(f'Expected unique restore before target {target_id}, got {len(candidates)}')
        if not candidates:
            continue
        tid, job = candidates[0]
        post, = job['single']
        sent, = job['send']
        start = job['process'][-1]
        reported = float(r['restore_server_ms'])
        traced = sent - start
        # The final dispatch must succeed and agree with the independent API timer.
        if not post <= start <= sent or abs(traced-reported) > 2 or any(t >= start for t in job['decline']):
            raise ValueError(f'Callback disagrees with the API timer for task {tid}: {traced}, {reported}')
        elapsed = float(r['restore_end_ms']) - float(r['restore_start_ms'])
        remainder = elapsed - (sent-post)
        if remainder < -.1:
            raise ValueError('Server span exceeds host API span')
        markers = [(stamp, cells) for stamp, cells in metadata if start <= stamp <= sent]
        if len(markers) != 1 or markers[0][1] != int(r['boundary']):
            raise ValueError(f'Expected one matching restore metadata event for task {tid}, got {markers}')
        marker = markers[0][0]
        # Cached-position log is a server marker, not a device-completion event.
        peer_positions = positions[0, peer[3]]
        peer_pos = next((pos for ts, pos in reversed(peer_positions) if ts <= start), '')
        next_pos_time, next_pos = next(((ts, pos) for ts, pos in peer_positions if ts >= sent), (None, ''))
        r.update(restore_task=tid, restore_queue_ms=round(start-post, 6), restore_server_ms_trace=round(traced, 6),
                 restore_dispatch_from_peer_post_ms=round(start-peer_post, 6),
                 restore_dispatch_from_peer_release_ms=round(start-releases[0, peer[3]], 6),
                 peer_pos_at_restore_dispatch=peer_pos,
                 peer_next_pos_after_restore=next_pos,
                 peer_next_pos_from_restore_result_ms=round(next_pos_time-sent, 6) if next_pos_time is not None else '',
                 restore_before_metadata_ms=round(marker-start, 6), restore_metadata_to_result_ms=round(sent-marker, 6),
                 restore_decoding_declines=len(job['decline']), restore_slot_defers=len(job['defer']),
                 restore_client_outside_server_ms=round(remainder, 6),
                 restore_result_to_target_queue_ms=round(target_post-sent, 6))


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        key = int(row['seed']), int(row['peer_total_tokens']), int(row['requested_lead_ms'])
        if 'storage_condition' in row:
            key += (row['storage_condition'],)
        if 'vulkan_timing' in row:
            key += (row['vulkan_timing'],)
        groups[key].append(row)
    for key, group in sorted(groups.items()):
        def mid(k):
            return round(median(float(r[k]) for r in group), 2)

        result = f'{key} target queue / launch-to-batch = {mid("target_queue_ms")}/{mid("target_launch_to_batch_ms")} ms'
        idle = [float(r['target_idle_save_ms']) for r in group]
        result += f'; idle saves in {sum(v > 0 for v in idle)}/{len(group)}, max {max(idle):.2f} ms'
        if 'target_peer_checkpoint_ms' in group[0]:
            ckpt = [float(r['target_peer_checkpoint_ms']) for r in group]
            result += f'; peer checkpoint median {median(ckpt):.2f} ms, observed in {sum(v > 0 for v in ckpt)}/{len(group)}'
        if key[2] >= 0:
            queue, run, outer = (mid(k) for k in ('restore_queue_ms', 'restore_server_ms_trace', 'restore_client_outside_server_ms'))
            declined = sum(int(r['restore_decoding_declines']) > 0 for r in group)
            deferred = sum(int(r['restore_slot_defers']) > 0 for r in group)
            result += f'; restore queue / callback / outside = {queue}/{run}/{outer} ms; declined {declined}/{len(group)}, slot-deferred {deferred}'
            if 'restore_before_metadata_ms' in group[0]:
                result += f'; callback before / after metadata = {mid("restore_before_metadata_ms")}/{mid("restore_metadata_to_result_ms")} ms'
        print(result)


def warm_pairs(rows):
    pairs = defaultdict(dict)
    for r in rows:
        key = tuple(int(r[k]) for k in ('seed', 'peer_total_tokens', 'requested_lead_ms', 'rep'))
        media = r['storage_condition']
        size, disk = int(r['file_bytes']), int(r['volume_read_bytes'])
        if (media not in {'cold', 'warm'} or media in pairs[key] or int(r['boundary']) != 1152
                or not 0 < size == int(r['restored_bytes'])
                or not (size <= disk if media == 'cold' else 0 <= disk < (1 << 20))):
            raise ValueError(f'Unverified or duplicate media control {key}, {media}')
        pairs[key][media] = r
    groups = defaultdict(list)
    for key, pair in pairs.items():
        if set(pair) != {'cold', 'warm'}:
            raise ValueError(f'Unpaired media control {key}')
        groups[key[:-1]].append((pair['cold'], pair['warm']))
    for key, group in sorted(groups.items()):
        print(f'{key}: {len(group)} media pairs, first token agrees in '
              f'{sum(a["greedy_token"] == b["greedy_token"] for a,b in group)}')
        fields = ('volume_read_bytes', 'restore_queue_ms', 'restore_server_ms_trace',
                  'restore_before_metadata_ms', 'restore_metadata_to_result_ms', 'target_queue_ms',
                  'target_launch_to_batch_ms', 'target_finish_ms', 'peer_finish_ms')
        for k in (k for k in fields if k in group[0][0]):
            gains = [float(a[k])-float(b[k]) for a, b in group]
            print(f'  cold minus warm {k}: median={median(gains):+.2f}, min={min(gains):+.2f}, max={max(gains):+.2f}')


def matched_trials(before, after):
    def index(rows):
        keys = ('seed', 'rep', 'total_tokens', 'peer_total_tokens', 'boundary', 'requested_lead_ms', 'release_ms')
        result = {tuple(r[k] for k in keys): r for r in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate trial key')
        return result

    on, off = index(before), index(after)
    if on.keys() != off.keys():
        raise ValueError('Runs do not cover identical cases')
    return on, off


def compare_idle_save(before, after):
    on, off = matched_trials(before, after)
    if any(int(r['target_idle_save_starts']) or float(r['target_idle_save_ms']) for r in after):
        raise ValueError('Idle-save marker present in disabled run')
    agrees = sum(on[k]['greedy_token'] == r['greedy_token'] for k, r in off.items())
    print(f'Idle-save on/off: {len(on)} matched trials, first token agrees in {agrees}')
    for k, a in sorted(on.items(), key=lambda pair: float(pair[1]['target_idle_save_ms']), reverse=True):
        if float(a['target_idle_save_ms']) <= 0:
            break
        b = off[k]
        label = f'peer={a["peer_total_tokens"]}, lead={a["requested_lead_ms"]}, rep={a["rep"]}'
        print(f'  {label}: save={float(a["target_idle_save_ms"]):.3f} ms, launch-to-batch '
              f'{float(a["target_launch_to_batch_ms"]):.3f} -> {float(b["target_launch_to_batch_ms"]):.3f} ms')


def compare_checkpoints(before, after):
    on, off = matched_trials(before, after)
    if any(float(r['target_peer_checkpoint_ms']) for r in after):
        raise ValueError('Peer checkpoint present in disabled run')
    agrees = sum(on[k]['greedy_token'] == r['greedy_token'] for k, r in off.items())
    print(f'Context checkpoints on/off: {len(on)} matched trials, first token agrees in {agrees}')
    groups = defaultdict(list)
    for k, a in on.items():
        b = off[k]
        if float(a['target_peer_checkpoint_ms']) > 0:
            groups[int(a['peer_total_tokens']), int(a['requested_lead_ms'])].append((a, b))
    for key, group in sorted(groups.items()):
        before_mid, after_mid = (median(float(pair[i]['target_launch_to_batch_ms']) for pair in group) for i in (0, 1))
        print(f'  {key}: {len(group)} trials with peer checkpoint in baseline, '
              f'launch-to-batch median {before_mid:.2f} -> {after_mid:.2f} ms')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--warm-pairs', action='store_true')
    parser.add_argument('--idle-save-off', type=Path, help='Compare with a matched archive with idle-slot saving disabled')
    parser.add_argument('--checkpoints-off', type=Path, help='Compare with a matched archive with context checkpoints disabled')
    args = parser.parse_args()
    with args.csv.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if args.log:
        extract(rows, args.log)
    rows = [r for r in rows if int(r['rep']) >= 0]
    if not rows:
        parser.error('No measured trials')
    summarize(rows)
    if args.warm_pairs:
        warm_pairs(rows)
    if args.idle_save_off:
        with args.idle_save_off.open(encoding='utf-8', newline='') as stream:
            compare_idle_save(rows, list(csv.DictReader(stream)))
    if args.checkpoints_off:
        with args.checkpoints_off.open(encoding='utf-8', newline='') as stream:
            compare_checkpoints(rows, list(csv.DictReader(stream)))
    if args.out:
        with args.out.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
