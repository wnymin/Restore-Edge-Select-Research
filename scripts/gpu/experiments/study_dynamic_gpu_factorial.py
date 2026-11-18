"""Measure separate and joint peer delays from restore and target work."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import re
from statistics import mean, median

from scripts.gpu.experiments.study_dynamic_gpu_restore_queue import read_trace
from scripts.gpu.analysis.summarize_dynamic_gpu_delay import engine_events


ARMS = ('peer', 'restore', 'target', 'both')
RESTORE = {'restore', 'both'}
TARGET = {'target', 'both', 'phase_target'}
POSITION = re.compile(r' id +([01]) [|] task ([0-9]+) [|] cached n_tokens = ([0-9]+),')


def batches_by_position(log):
    """Find slot contributions grouped before the same server decode event."""
    pending, by_task = {}, defaultdict(list)
    with log.open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            if m := POSITION.search(line):
                slot, task, cached = map(int, m.groups())
                pending[slot, task] = cached
            if ' D srv        decode: n_batch (effective)' in line and 'off = 0' in line and pending:
                group = pending.copy()
                for task in group:
                    by_task[task].append(group)
                pending.clear()
    return by_task


def extract(rows, log):
    lengths = {int(r[k]) for r in rows for k in ('total_tokens', 'peer_total_tokens')}
    batches, releases, positions = engine_events(log, lengths)
    scheduled = batches_by_position(log)
    events, _, idle_starts, checkpoints, metadata, caps = read_trace(log)
    if idle_starts or any(checkpoints.values()):
        raise ValueError('Unexpected idle save or peer context checkpoint')
    if any(caps[k] != {int(r[field]) for r in rows}
           for k, field in (('n_batch', 'batch_tokens'), ('n_ubatch', 'ubatch_tokens'))):
        raise ValueError('Unexpected actual batch caps')
    starts = [b for b in batches if b[0] == 0]
    targets = iter(b for b in batches if b[0] == 1)
    if len(starts) != len(rows):
        raise ValueError('Expected one peer batch per trial, including warmup')
    candidates = {k: v for k, v in events.items() if v['wait'] and v['single'] and v['send']}
    for r, peer in zip(rows, starts, strict=True):
        arm, peer_id = r['arm'], peer[3]
        if peer[:2] != (0, 512) or peer[4] != int(r['peer_total_tokens']):
            raise ValueError('Invalid peer cached position or total length')
        post, = events[peer_id]['post']
        batch, done = peer[2], releases[0, peer_id]
        if not post <= batch <= done:
            raise ValueError('Invalid peer event ordering')
        r.update(peer_task=peer_id, peer_server_ms=round(done-post, 6), peer_first_batch_from_post_ms=round(batch-post, 6),
                 target_task='', target_post_from_peer_post_ms='', target_batch_from_peer_post_ms='',
                 target_batch_to_peer_release_ms='', target_batch_to_release_ms='', target_first_batch_has_peer='',
                 peer_cached_in_target_first_batch='', shared_prefill_batches='', restore_task='', restore_queue_ms='',
                 restore_dispatch_from_peer_post_ms='', restore_before_metadata_ms='', restore_after_metadata_ms='',
                 restore_result_from_peer_post_ms='', peer_pos_at_restore='', peer_next_pos_after_restore='',
                 peer_batch2_from_post_ms='', peer_final_prefill_cached='', peer_final_prefill_to_release_ms='')
        peer_positions = positions[0, peer_id]
        last_prefill, last_cached = peer_positions[-1]
        r.update(peer_final_prefill_cached=last_cached, peer_final_prefill_to_release_ms=round(done-last_prefill, 6))
        if len(peer_positions) >= 2:
            r['peer_batch2_from_post_ms'] = round(peer_positions[1][0]-post, 6)
        target_post = None
        if arm in TARGET:
            target = next(targets, None)
            if (not target or target[:2] != (1, int(r['boundary'])) or target[4] != int(r['total_tokens'])):
                raise ValueError('Invalid target cached position or total length')
            target_id = target[3]
            target_post, = events[target_id]['post']
            target_groups = scheduled[1, target_id]
            if not target_groups:
                raise ValueError('No target prefill position linked to decode')
            if not post <= target_post <= target[2] <= releases[1, target_id]:
                raise ValueError('Invalid target event ordering')
            r.update(target_task=target_id, target_post_from_peer_post_ms=round(target_post-post, 6),
                     target_batch_from_peer_post_ms=round(target[2]-post, 6),
                     target_batch_to_peer_release_ms=round(done-target[2], 6),
                     target_batch_to_release_ms=round(releases[1, target_id]-target[2], 6),
                     target_first_batch_has_peer=int((0, peer_id) in target_groups[0]),
                     peer_cached_in_target_first_batch=target_groups[0].get((0, peer_id), ''),
                     shared_prefill_batches=sum((0, peer_id) in group for group in target_groups))
        # No other single HTTP result can occur before peer release: the next trial starts after both arms finish.
        jobs = [(tid, e) for tid, e in candidates.items() if post < e['single'][0] <= e['send'][-1] <= done]
        if len(jobs) != int(arm in RESTORE):
            raise ValueError(f'Expected a restore before peer release only in restore arms: {peer_id=}, {arm=}')
        if jobs:
            tid, job = jobs[0]
            enqueue, = job['single']
            sent, = job['send']
            run = job['process'][-1]
            if (not enqueue <= run <= sent or (target_post is not None and sent > target_post)
                    or any(t >= run for t in job['decline']) or abs(sent-run-float(r['restore_server_ms'])) > 2):
                raise ValueError(f'Restore dispatch does not match API response {tid}')
            (t, cells), = [(t, n) for t, n in metadata if run <= t <= sent]
            if cells != int(r['boundary']):
                raise ValueError('Restore metadata does not match deep prefix')
            r.update(restore_task=tid, restore_queue_ms=round(run-enqueue, 6),
                     restore_dispatch_from_peer_post_ms=round(run-post, 6),
                     restore_before_metadata_ms=round(t-run, 6), restore_after_metadata_ms=round(sent-t, 6),
                     restore_result_from_peer_post_ms=round(sent-post, 6),
                     peer_pos_at_restore=next((n for ts,n in reversed(peer_positions) if ts <= run), ''),
                     peer_next_pos_after_restore=next((n for ts,n in peer_positions if ts >= sent), ''))
    if next(targets, None):
        raise ValueError('Unused target task in server log')


def analyze(rows):
    blocks = defaultdict(dict)
    phase = any(r['arm'] == 'phase_target' for r in rows)
    arm_names = ARMS + (('phase_target',) if phase else ())
    settings = {(r['batch_tokens'], r['ubatch_tokens'], r['release_ms'], r['lead_ms'], r['boundary']) for r in rows}
    if len(settings) != 1:
        raise ValueError('Expected one common configuration')
    for r in rows:
        key = (int(r['seed']), int(r['peer_total_tokens']), int(r['rep']))
        if r['arm'] not in arm_names or r['arm'] in blocks[key]:
            raise ValueError(f'Unknown or duplicate arm in {key}')
        blocks[key][r['arm']] = r
    groups = defaultdict(list)
    for k, arms in sorted(blocks.items()):
        if (arms.keys() != set(arm_names) or len({a['peer_token'] for a in arms.values()}) != 1
                or len({a['target_token'] for k,a in arms.items() if k in TARGET}) != 1):
            raise ValueError(f'Incomplete block or unequal greedy output in {k}')
        groups[k[:2]].append([arms[a] for a in arm_names])
    for key, group in sorted(groups.items()):
        def values(field, idx=None):
            if idx is not None:
                return [float(arms[idx][field]) for arms in group]
            return [[float(r[field]) for r in arms[:4]] for arms in group]

        def show(label, vals):
            print(f'  {label}: mean {mean(vals):+.2f}, median {median(vals):+.2f}, '
                  f'range [{min(vals):+.2f},{max(vals):+.2f}] ms; positive {sum(x > 0 for x in vals)}/{len(vals)}')

        print(f'{key}: {len(group)} {len(arm_names)}-arm blocks with matching first tokens')
        for metric in ('peer_finish_ms', 'peer_server_ms', 'peer_batch2_from_post_ms'):
            print(metric)
            x = values(metric)
            show('restore only minus peer only', [r-p for p,r,t,b in x])
            show('target only minus peer only', [t-p for p,r,t,b in x])
            show('both minus peer only', [b-p for p,r,t,b in x])
            show('both minus target only', [b-t for p,r,t,b in x])
            show('both minus restore only', [b-r for p,r,t,b in x])
            show('interaction', [b-r-t+p for p,r,t,b in x])
        for i in (1, 3):
            count = sum(float(a[i]['restore_end_ms']) <= float(a[i]['release_ms']) for a in group)
            print(f'  {ARMS[i]} ready at release {count}/{len(group)}')
            for field in ('restore_queue_ms', 'restore_dispatch_from_peer_post_ms', 'restore_before_metadata_ms',
                          'restore_after_metadata_ms', 'restore_result_from_peer_post_ms'):
                print(f'  {ARMS[i]} {field} median {median(values(field, i)):.2f} ms')
        for i in (2, 3):
            shared_first = sum(a[i]['target_first_batch_has_peer'] in (1, '1') for a in group)
            print(f'  {ARMS[i]} target and peer in first target prefill: {shared_first}/{len(group)}')
            for field in ('target_submit_ms', 'target_post_from_peer_post_ms', 'target_batch_from_peer_post_ms',
                          'target_batch_to_peer_release_ms', 'target_batch_to_release_ms', 'target_finish_ms'):
                print(f'  {ARMS[i]} {field} median {median(values(field, i)):.2f} ms')
        show('target response, both minus target only', [b-t for t,b in zip(values('target_finish_ms', 2),
                                                                           values('target_finish_ms', 3))])
        if phase:
            aligned = []
            for arms in group:
                both, probe = arms[3:]
                pos = both['peer_cached_in_target_first_batch']
                if (pos != '' and pos == probe['peer_cached_in_target_first_batch']
                        and all(r['peer_final_prefill_cached'] == pos for r in arms)):
                    aligned.append(arms)
            print(f'  same peer position in first target batch and at final peer prefill: {len(aligned)}/{len(group)}')
            show('phase target minus peer only (all trials, peer client)',
                 [float(a[4]['peer_finish_ms'])-float(a[0]['peer_finish_ms']) for a in group])
            if aligned:
                for field in ('peer_server_ms', 'peer_final_prefill_to_release_ms'):
                    print(f'  {field} (phase-matched only)')
                    x = [[float(r[field]) for r in a] for a in aligned]
                    show('phase target minus peer only', [q-p for p,r,t,b,q in x])
                    show('both minus restore only', [b-r for p,r,t,b,q in x])
                    show('difference of those contrasts', [(b-r)-(q-p) for p,r,t,b,q in x])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    with args.csv.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if args.log:
        extract(rows, args.log)
    rows = [r for r in rows if int(r['rep']) >= 0]
    analyze(rows)
    if args.out:
        with args.out.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator=chr(10))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
