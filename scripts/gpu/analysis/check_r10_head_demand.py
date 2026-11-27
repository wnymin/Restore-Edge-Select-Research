"""Audit direct-to-active FIFO head demand when the prefetch arena is too small."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
from itertools import groupby
import json
from pathlib import Path

PAGE = 65536
DEPTH_DEPS = {512: (0, 1), 1024: (0, 2, 3)}


def audit(path):
    groups = defaultdict(dict)
    formal = Counter()
    cases = {}
    with path.open('rb') as origin:
        byte_order = origin.read(2)
    with path.open(newline='', encoding='utf-16' if byte_order in (b'\xff\xfe', b'\xfe\xff') else 'utf-8-sig') as f:
        records = csv.DictReader(f)
        for (rep, arm), group in groupby(records, lambda r: (int(r['rep']), r['arm'])):
            rows = [{k: v if k in ('arm', 'event') else int(v) for k, v in r.items()} for r in group]
            mode, depth = arm.split('-')
            assert mode in ('staged', 'cold')
            depth = int(depth)
            cold = mode == 'cold'
            deps = DEPTH_DEPS[depth]

            def matching(event):
                return [r for r in rows if r['event'] == event]

            def one(event):
                row, = matching(event)
                return row

            assert [r['event_no'] for r in rows] == list(range(len(rows)))
            assert [r['wall_us'] for r in rows] == sorted(r['wall_us'] for r in rows)
            sizes = {r['object']: r['bytes'] for r in matching('catalog-size')}
            assert len(sizes) == 4 and all(n % PAGE == 0 and n > 0 for n in sizes.values())
            need = sum(sizes[i] for i in deps)
            arena = one('device-arena')['bytes']
            assert arena == need - (PAGE if cold else 0)
            assert all(r['capacity_bytes'] == arena and 0 <= r['charged_bytes'] <= arena for r in rows)
            start = one('direct-head-demand' if cold else 'reserve')
            assert start['request'] == 1
            if cold:
                rejected = one('reject')
                assert rejected['event_no'] < start['event_no'] and start['bytes'] == depth
                assert not any(matching(e) for e in ('reserve', 'device-copy', 'release', 'upload-submit', 'upload-ready'))
                assert all(r['charged_bytes'] == 0 for r in rows)
            else:
                assert start['charged_bytes'] == arena and all(not matching(e) for e in ('reject', 'direct-head-demand'))
                assert one('device-copy')['bytes'] == arena and one('release')['charged_bytes'] == 0
            sent = matching('direct-upload-submit' if cold else 'upload-submit')
            ready = matching('direct-upload-ready' if cold else 'upload-ready')
            assert [r['object'] for r in sent] == list(deps)
            last = start['event_no']
            last_ticket = 0
            for source, dest in zip(sent, ready, strict=True):
                assert last < source['event_no'] < dest['event_no']
                assert all(source[k] == dest[k] for k in ('fence', 'object', 'bytes'))
                assert source['fence'] > last_ticket and source['bytes'] == sizes[source['object']]
                if cold:
                    assert source['request'] == dest['request'] == 1
                last, last_ticket = dest['event_no'], dest['fence']
            disk = one('disk-read')
            assert disk['object'] == deps[-1] and disk['bytes'] == sizes[deps[-1]]
            assert sent[-2]['event_no'] < disk['event_no'] < sent[-1]['event_no']
            prefix, consumed, end = map(one, ('prefix-work', 'consumed', 'case-end'))
            assert last < prefix['event_no'] < consumed['event_no'] < end['event_no']
            assert prefix['bytes'] == 1024 - depth and consumed['status'] == 1 and consumed['bytes'] == 8
            assert end['charged_bytes'] == 0
            assert mode not in groups[(rep, depth)]
            groups[(rep, depth)][mode] = (need, disk['bytes'])
            if rep >= 0:
                formal[arm] += 1
                observed = dict(device_arena_bytes=arena, peak_charged_bytes=max(r['charged_bytes'] for r in rows),
                                disk_bytes=disk['bytes'], h2d_bytes=need,
                                device_copy_bytes=0 if cold else one('device-copy')['bytes'])
                assert arm not in cases or cases[arm] == observed
                cases[arm] = observed
    assert set(formal) == {'cold-512', 'staged-512', 'cold-1024', 'staged-1024'}
    assert len(set(formal.values())) == 1
    for variants in groups.values():
        assert variants.keys() == {'cold', 'staged'} and variants['cold'] == variants['staged']
    return dict(scope='idle-head exact direct demand; competing later reservations excluded',
                cases=cases, arms=dict(formal), exact_logit_positions=8 * sum(formal.values()),
                raw_sha256=sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    output = json.dumps(audit(args.input), indent=2, sort_keys=True)
    if args.out:
        with args.out.open('x', encoding='utf-8') as f:
            print(output, file=f)
    print(output)


if __name__ == '__main__':
    main()
