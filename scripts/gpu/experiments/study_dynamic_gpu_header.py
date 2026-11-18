"""Check whether in-flight work delays slot restoration before reading a valid payload."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import median

from scripts.gpu.experiments.study_dynamic_gpu_restore_queue import STAMP, read_trace
from scripts.gpu.analysis.summarize_dynamic_gpu_delay import engine_events


HEADER_ERROR = 'E state_seq_load_file: unknown (magic, version) for sequence state file: 00000000, 00000000'
CONDITIONS = {(p, f) for p in ('active', 'released') for f in ('header', 'warm', 'cold')}


def extract(rows, log):
    peers, releases, positions = engine_events(log, {int(r['peer_total_tokens']) for r in rows})
    events, _, idle, checkpoints, metadata, caps = read_trace(log)
    if idle or any(checkpoints.values()) or len(peers) != len(rows):
        raise ValueError('Expected one peer and no automatic cache saves per trial')
    if any(caps[k] != {int(r[field]) for r in rows}
           for k, field in (('n_batch', 'batch_tokens'), ('n_ubatch', 'ubatch_tokens'))):
        raise ValueError('Unexpected actual batch caps')
    markers = [(t, 'metadata', cells) for t, cells in metadata]
    with log.open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            if HEADER_ERROR in line:
                m, s, ms, us = map(int, STAMP.match(line).groups())
                markers.append((m*60000+s*1000+ms+us/1000, 'header', 0))
    calls = []
    for tid, job in events.items():
        if not (job['single'] and job['wait'] and job['send'] and job['process']):
            continue
        start, result = job['process'][-1], job['send'][-1]
        hit = [(t, kind, cells) for t, kind, cells in markers if start <= t <= result]
        if not hit:
            continue
        (t, kind, cells), = hit
        post, = job['single']
        sent, = job['send']
        if not post <= start <= t <= sent or any(v >= start for v in job['decline']):
            raise ValueError(f'Invalid restoration events for {tid}')
        calls.append((post, start, t, sent, kind, cells, tid, len(job['decline'])))
    calls.sort()
    if len(calls) != len(rows) or len(markers) != len(rows):
        raise ValueError('Unmatched file read, including warmup')
    for r, peer, call in zip(rows, peers, calls, strict=True):
        enqueue, start, mark, sent, kind, cells, tid, declined = call
        peer_id = peer[3]
        peer_post, = events[peer_id]['post']
        peer_done = releases[0, peer_id]
        active = r['peer_condition'] == 'active'
        if (peer[:2] != (0, 512) or peer[4] != int(r['peer_total_tokens']) or not peer_post <= enqueue
                or active != (sent < peer_done)):
            raise ValueError(f'Expected verifiable active or released restoration: {peer_id=}, {tid=}')
        if kind == 'header':
            if r['file_condition'] != 'header' or int(r['http_status']) != 400:
                raise ValueError('Invalid rejection control')
        elif r['file_condition'] == 'header' or cells != int(r['boundary']) or abs(sent-start-float(r['restore_server_ms'])) > 2:
            raise ValueError('Successful restore disagrees with the reported callback or prefix')
        host_span = float(r['restore_end_ms']) - float(r['restore_start_ms'])
        if host_span < sent-enqueue-0.1:
            raise ValueError('Server interval exceeds the HTTP call')
        r.update(peer_task=peer_id, restore_task=tid, observed_marker=kind, restore_queue_ms=round(start-enqueue, 6),
                 dispatch_to_marker_ms=round(mark-start, 6), marker_to_result_ms=round(sent-mark, 6),
                 dispatch_to_result_ms=round(sent-start, 6), restore_dispatch_from_peer_post_ms=round(start-peer_post, 6),
                 peer_cached_at_dispatch=next((pos for ts,pos in reversed(positions[0, peer_id]) if ts <= start), ''),
                 restore_declines=declined)


def analyze(rows):
    blocks = defaultdict(dict)
    for r in rows:
        key = (int(r['seed']), int(r['peer_total_tokens']), int(r['rep']))
        cond = (r['peer_condition'], r['file_condition'])
        size, disk, restored = (int(r[k]) for k in ('file_bytes', 'volume_read_bytes', 'restored_bytes'))
        if cond not in CONDITIONS or cond in blocks[key] or not 0 <= disk:
            raise ValueError(f'Unverified or repeated condition {key}, {cond}')
        if cond[1] == 'header':
            if (size, restored, r['http_status'], r['observed_marker']) != (8, 0, '400', 'header') or disk >= 1 << 20:
                raise ValueError('Expected a RAM-cached invalid header without any restored payload')
        elif not 0 < size == restored or r['observed_marker'] != 'metadata' or (disk >= size) != (cond[1] == 'cold'):
            raise ValueError('Expected independently verified successful warm or cold restore')
        blocks[key][cond] = r
    groups = defaultdict(list)
    for key, arms in blocks.items():
        if arms.keys() != CONDITIONS or len({r['peer_token'] for r in arms.values()}) != 1:
            raise ValueError(f'Missing arm or changed peer token {key}')
        groups[key[:2]].append(arms)
    metrics = ('restore_queue_ms', 'dispatch_to_marker_ms', 'marker_to_result_ms', 'dispatch_to_result_ms')
    for key, blocks in sorted(groups.items()):
        print(f'{key}: {len(blocks)} paired blocks, first token agrees')
        for cond in sorted(CONDITIONS):
            print(cond, ', '.join(f'{m}={median(float(b[cond][m]) for b in blocks):.3f}' for m in metrics))
        for f in ('header', 'warm', 'cold'):
            diffs = [float(b['active', f]['dispatch_to_marker_ms'])-float(b['released', f]['dispatch_to_marker_ms']) for b in blocks]
            print(f'{f} active minus released to marker: {median(diffs):+.3f} ms [{min(diffs):+.3f}, {max(diffs):+.3f}]')
        deltas = [float(b['active', 'warm']['dispatch_to_marker_ms'])-float(b['active', 'header']['dispatch_to_marker_ms'])
                  for b in blocks]
        print(f'active warm metadata minus invalid header: {median(deltas):+.3f} ms [{min(deltas):+.3f}, {max(deltas):+.3f}]')


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
    if not rows:
        parser.error('No measurement rows')
    analyze(rows)
    if args.out:
        with args.out.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator=chr(10))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
