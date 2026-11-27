"""Check active KV borrowing against same-input upload-and-copy controls."""

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
from itertools import groupby
import json
from pathlib import Path


def audit(path):
    with path.open('rb') as f:
        encoding = 'utf-16' if f.read(2).hex() in ('fffe', 'feff') else 'utf-8-sig'
    rounds = defaultdict(dict)
    with path.open(encoding=encoding, newline='') as f:
        for (rep, arm), raw in groupby(csv.DictReader(f), lambda r: (int(r['rep']), r['arm'])):
            assert arm in ('active-borrow', 'active-copy-control')
            borrow = arm == 'active-borrow'
            rows = [{k: v if k in ('arm', 'event') else int(v) for k, v in r.items()} for r in raw]
            groups = defaultdict(list)
            for i, r in enumerate(rows):
                assert r['event_no'] == i and 0 <= r['charged_bytes'] <= r['capacity_bytes'] == 86245376
                groups[r['event']].append(r)
            submit = groups['upload-submit']
            assert [r['object'] for r in submit] == ([0, 1, 2, 3] if borrow else [0, 1, 0, 2, 3])
            last = -1
            for sent, ready in zip(submit, groups['upload-ready'], strict=True):
                assert last < sent['event_no'] < ready['event_no']
                assert (sent['object'], sent['bytes'], sent['fence']) == (ready['object'], ready['bytes'], ready['fence'])
                last = ready['event_no']
            donor, = groups['borrow-donor-query']
            assert donor['status'] in (0, 1) and donor['fence'] > 0
            first, second = groups['reserve']
            assert first['request'] == 0 and second['request'] == 1 and second['charged_bytes'] == 86245376
            assert groups['release'][0]['charged_bytes'] == donor['charged_bytes'] == 0
            assert groups['release'][0]['event_no'] < groups['first-suffix-submitted'][0]['event_no'] < donor['event_no'] < second['event_no']
            if borrow:
                alias, = groups['active-kv-borrow']
                reused, = groups['active-kv-reused']
                assert alias['bytes'] == reused['bytes'] == 16777216 and alias['object'] == reused['object'] == 0
                assert alias['refs'] == 1 and alias['charged_bytes'] == 86245376
                assert second['event_no'] < alias['event_no'] < reused['event_no']
            else:
                assert not groups['active-kv-borrow'] and not groups['active-kv-reused']
            assert [r['object'] for r in groups['disk-read']] == [1, 3]
            assert [(r['request'], r['bytes'], r['status']) for r in groups['consumed']] == [(0, 8, 1), (1, 8, 1)]
            assert rows[-1]['event'] == 'case-end' and rows[-1]['charged_bytes'] == 0
            assert arm not in rounds[rep]
            rounds[rep][arm] = (sum(r['bytes'] for r in submit), groups['device-copy'][1]['bytes'], int(donor['status'] == 0))
    formal = [arms for rep, arms in rounds.items() if rep >= 0]
    assert formal and all(set(r) == {'active-borrow', 'active-copy-control'} for r in rounds.values())
    delta = [[a - b for a, b in zip(r['active-copy-control'][:2], r['active-borrow'][:2], strict=True)] for r in formal]
    assert all(d == [16777216, 16777216] for d in delta)
    return dict(pairs=len(formal), logit_positions=len(formal) * 32,
                per_pair_ram_upload_avoided_bytes=delta[0][0], per_pair_device_copy_avoided_bytes=delta[0][1],
                prior_first_suffix_fence_pending=sum(v[2] for r in formal for v in r.values()),
                raw_sha256=sha256(path.read_bytes()).hexdigest(),
                scope='same active tensor addresses, immediate successor, forced boundary, no extra device materialization')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('raw', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    result = json.dumps(audit(a.raw), sort_keys=True, indent=2)
    if a.out:
        with a.out.open('x', encoding='utf-8') as f:
            print(result, file=f)
    print(result)


if __name__ == '__main__':
    main()
