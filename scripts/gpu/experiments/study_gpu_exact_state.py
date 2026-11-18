"""Check round-trip correctness and exact-checkpoint staging quota comparisons."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import median


def read(path):
    with path.open(encoding='utf-8', newline='') as f:
        return [r for r in csv.DictReader(f) if int(r['rep']) >= 0]


def summarize_exact(rows):
    groups = defaultdict(dict)
    for r in rows:
        block = groups[int(r['rep']), int(r['depth'])]
        if r['arm'] in block or r['token_match'] != '1' or float(r['max_logits_error']) > 1e-6:
            raise ValueError(f'Round trip failed: {r}')
        block[r['arm']] = r
    for (rep, depth), arms in groups.items():
        if arms.keys() != {'host', 'device', 'file'}:
            raise ValueError(f'Incomplete block: {(rep, depth)}')
        if len({(r['host_state_bytes'], r['device_metadata_bytes'], r['file_bytes']) for r in arms.values()}) != 1:
            raise ValueError('State sizes changed between paths')
    for depth in sorted({k[1] for k in groups}):
        blocks = [arms for k, arms in groups.items() if k[1] == depth]
        print(f'{depth} tokens: {len(blocks)} exact blocks; restore median ms:',
              {arm: round(median(float(b[arm]['restore_ms']) for b in blocks), 3) for arm in ('host', 'device', 'file')})


def summarize_stage(rows):
    blocks = defaultdict(dict)
    validation = [r for r in rows if int(r['rep']) < 0]
    if len(validation) != 6 or any(r.get('state_bytes_equal') != '1' for r in validation):
        raise ValueError('Need verified pre-continuation serialized state equality for all six cases')
    rows = [r for r in rows if int(r['rep']) >= 0]
    depths = sorted({int(r['depth']) for r in rows})
    if len(depths) != 2 or depths[0] != 1152:
        raise ValueError(f'Invalid stage depths: {depths}')
    for r in rows:
        rep, cap, depth = [int(r[k]) for k in ('rep', 'quota_mib', 'depth')]
        target, peer, read_bytes, peak, waited = [int(r[k]) for k in
            ('target_object_bytes', 'peer_object_bytes', 'read_bytes', 'peak_bytes', 'waited')]
        shared = r['topology'] == 'shared'
        expect = int(shared and target + peer > cap * (1 << 20))
        enqueue, release, peer_work, consume, done = [float(r[k]) for k in
            ('target_stage_enqueue_ms', 'peer_release_ms', 'peer_compute_ms', 'target_consume_ms', 'target_ms')]
        if (expect != waited or peak != (max(peer, target) if waited else peer + target)
                or max(peer, target) > cap * (1 << 20) or read_bytes % 4096 or not peer <= read_bytes < peer + 4096
                or waited and enqueue + .01 < release + peer_work
                or not waited and enqueue > release or not release <= consume <= done):
            raise ValueError(f'Staging contract failed: {r}')
        key = r['topology'], cap, rep
        if depth in blocks[key]:
            raise ValueError(f'Duplicate: {key}')
        blocks[key][depth] = r
    cases = defaultdict(list)
    for (topology, cap, rep), arms in blocks.items():
        if arms.keys() != set(depths):
            raise ValueError(f'Incomplete block: {(topology, cap, rep)}')
        s, d = (arms[depth] for depth in depths)
        cases[topology, cap].append((float(d['target_ms']) - float(s['target_ms']), int(d['waited']),
                                    float(d['target_stage_enqueue_ms']) - float(s['target_stage_enqueue_ms'])))
    for case, values in sorted(cases.items()):
        d, w, q = zip(*values)
        print(f'{case}: {len(values)} pairs; deep D-S median {median(d):+.3f} ms; '
              f'deep wins {sum(x < 0 for x in d)}; deep waited {sum(w)}; '
              f'deep enqueue delay relative to shallow {median(q):+.3f} ms')
    print(f'Staged greedy match {sum(r["top_token_match"] == "1" for r in rows)}/{len(rows)}; '
          f'max downstream logit deviation {max(float(r["max_logits_error"]) for r in rows):.6f}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exact', type=Path)
    p.add_argument('--stage', type=Path)
    args = p.parse_args()
    if not (args.exact or args.stage):
        p.error('Provide at least one evidence file')
    for path, verify in ((args.exact, summarize_exact), (args.stage, summarize_stage)):
        if path:
            if path == args.stage:
                with path.open(encoding='utf-8', newline='') as f:
                    rows = list(csv.DictReader(f))
            else:
                rows = read(path)
            if not rows:
                p.error(f'No measured blocks in {path}')
            verify(rows)


if __name__ == '__main__':
    main()
