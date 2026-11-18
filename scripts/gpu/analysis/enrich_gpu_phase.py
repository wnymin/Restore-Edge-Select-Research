"""Tie GPU phase trials to server task IDs, restore phases and shared batches."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import re

from scripts.gpu.experiments.study_dynamic_gpu_restore_queue import read_trace
from scripts.gpu.analysis.summarize_dynamic_gpu_delay import engine_events


POSITION = re.compile(r' id\s+([01]) \| task (\d+) \| cached n_tokens = (\d+),')
TOKEN = re.compile(r' id\s+([01]) \| task (\d+) \| n_gen = (\d+),')


def stamp(text):
    m, s, ms, us = map(int, text.split('.'))
    return m * 60000 + s * 1000 + ms + us / 1000


def shared(log):
    pending, prefills, decodes, tokens = {}, {}, defaultdict(set), defaultdict(list)
    batch = None
    with log.open(encoding='utf-8', errors='replace') as f:
        for line in f:
            if m := POSITION.search(line):
                slot, task, count = map(int, m.groups())
                pending[slot, task] = count
            if 'D srv        decode: n_batch (effective)' in line and 'off = 0' in line:
                batch = stamp(line.split()[0])
                prefills[batch] = pending
                pending = {}
            if m := TOKEN.search(line):
                slot, task, count = map(int, m.groups())
                decodes[batch].add((slot, task))
                tokens[slot, task].append((count, stamp(line.split()[0])))
    return prefills, decodes, tokens


def enrich(rows, log, peer_log=None):
    isolated = {r.get('peer_executor', 'same') for r in rows} == {'isolated'}
    if isolated != (peer_log is not None):
        raise ValueError('Isolated captures require their second server log')
    lengths = {int(r[k]) for r in rows for k in ('target', 'peer') if int(r[k])}
    batches, releases, positions = engine_events(log, lengths)
    events, _, idle_saves, checkpoints, metadata, caps = read_trace(log)
    if idle_saves or any(checkpoints.values()) or caps['n_batch'] != {1024} or caps['n_ubatch'] != {512}:
        raise ValueError('Different engine settings from the phase protocol')
    prefills, decodes, tokens = shared(log)
    targets = [b for b in batches if b[0] == 1]
    peer_batches = batches
    if isolated:
        peer_batches, _, positions = engine_events(peer_log, lengths)
        _, _, tokens = shared(peer_log)
    peers = {b[3]: b for b in peer_batches if b[0] == 0}
    singles = sorted((v for v in events.values() if v['single'] and v['wait'] and len(v['send']) == 1),
                     key=lambda v: v['send'][0])
    previous_target_release = float('-inf')
    for row, target in zip(rows, targets, strict=True):
        target_id = target[3]
        post, = events[target_id]['post']
        launch, = events[target_id]['launch']
        arm = row.get('execution', row['arm'])
        depth = int(row['shallow' if arm == 'shallow' else 'deep'])
        if target[1] != depth or target[4] != int(row['target']) or not post <= launch <= target[2]:
            raise ValueError(f'Target event mismatch: {row}, {target}')
        row.update(target_task=target_id, target_queue_ms=round(launch-post, 3),
                   target_launch_to_batch_ms=round(target[2]-launch, 3),
                   peer_prefilling_in_target_first_batch=0, peer_decoding_in_target_first_batch=0,
                   peer_gap_srv_ms='', gate_to_dispatch_ms='', gate_to_target_batch_ms='',
                   restore_queue_ms='', restore_before_metadata_ms='', restore_after_metadata_ms='')
        gate_ts = None
        if row['peer_task']:
            peer_id = int(row['peer_task'])
            peer_key = (0, peer_id)
            peer = peers.pop(peer_id)
            gate_ts = stamp(row['server_gate'])
            step = int(row.get('gate_step') or {'early': 512, 'late': 2560, 'decode': 8}[row['phase']])
            expected = (step, gate_ts) if row['phase'] == 'decode' else (gate_ts, step)
            if (peer[1] != 512 or peer[4] != int(row['peer']) or gate_ts < peer[2] - 1
                    or not isolated and gate_ts > post
                    or expected not in (tokens[peer_key] if row['phase'] == 'decode' else positions[peer_key])):
                raise ValueError(f'Peer event mismatch: {row}, {peer}')
            ts = [t for _, t in tokens[peer_key]]
            if len(ts) != int(row['decode_n']):
                raise ValueError('Server and client disagree on number of peer tokens')
            pf = not isolated and peer_key in prefills[target[2]]
            row.update(peer_prefilling_in_target_first_batch=int(pf),
                       peer_decoding_in_target_first_batch=int(not isolated and not pf and peer_key in decodes[target[2]]),
                       peer_gap_srv_ms=round(max(b-a for a,b in zip(ts, ts[1:])), 3),
                       gate_to_target_batch_ms=round(target[2]-gate_ts, 3) if not isolated else '')
        if arm == 'deep':
            matches = [(v, ts) for v in singles if previous_target_release < v['single'][0] <= v['send'][0] <= post
                       for ts, n in metadata if v['process'][-1] <= ts <= v['send'][0] and n == depth]
            (job, read), = matches
            dispatch, send = job['process'][-1], job['send'][0]
            row.update(gate_to_dispatch_ms=round(dispatch-gate_ts, 3) if gate_ts is not None and not isolated else '',
                       restore_queue_ms=round(dispatch-job['single'][0], 3),
                       restore_before_metadata_ms=round(read-dispatch, 3), restore_after_metadata_ms=round(send-read, 3))
        previous_target_release = releases[1, target_id]
    if peers:
        raise ValueError(f'Unused peer tasks: {sorted(peers)}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    with args.capture.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    side = args.capture.with_suffix('.peer.log') if rows[0].get('peer_executor') == 'isolated' else None
    enrich(rows, args.capture.with_suffix('.log'), side)
    with args.out.open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
