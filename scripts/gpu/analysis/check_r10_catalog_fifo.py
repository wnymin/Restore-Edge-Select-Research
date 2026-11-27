"""Cross-check controller and native variable-catalogue diagnostic transcripts."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from scripts.validation.check_r10_native_catalog import digest


def check(run_file, native_file, schedule):
    run = json.loads(run_file.read_bytes())
    trace = json.loads(schedule.read_bytes())['trace']
    sizes = {int(o['id']): (o['bytes'], o['tier']) for o in trace['objects']}
    options = [{c['boundary_tokens']: tuple(map(int, c['load_order'])) for c in r['candidates']} for r in trace['requests']]
    decisions = {r['snapshot']['request']: r['action'] for r in run['decisions']}
    starts = [r['compute'] for r in run['starts']]
    n = len(options)
    assert len(run['decisions']) == n and set(decisions) == set(range(n))
    assert len(starts) == len(run['completions_ms']) == n
    deps = {}
    for i, start in enumerate(starts):
        assert start['request'] == i and start['boundary'] == decisions[i]
        deps[i] = options[i][decisions[i]]
        assert tuple(map(int, start['deps'])) == deps[i]
        if run['policy'] == 'deepest':
            assert decisions[i] == max(options[i])
    with native_file.open(encoding='ascii', newline='') as f:
        native = [dict(event=r.pop('event'), **{k: int(v) for k, v in r.items()}) for r in csv.DictReader(f)]
    counts = Counter(row['event'] for row in native)
    order = defaultdict(list)
    read, issued, observed, last_fence = None, None, [], 0
    hold_ms = run.get('ssd_hold_ms', 0)
    by_request = defaultdict(set)
    assert [r['native_us'] for r in native] == sorted(r['native_us'] for r in native)
    for row in native:
        event, req, ident = (row[k] for k in ('event', 'request', 'identity'))
        order[event].append(req)
        by_request[req].add(event)
        assert 0 <= row['charged_bytes'] <= run['capacity_bytes']
        if event in ('read-direct', 'read-staged', 'disk-complete', 'disk-soft-hold', 'upload-submit', 'upload-ready',
                     'native-scatter', 'carry-active-kv', 'preserve-old-active', 'preserve-recurrent'):
            assert row['bytes'] == sizes[ident][0]
        if event in ('native-target', 'suffix-submitted', 'reserved'):
            assert ident == decisions[req]
        if event in ('read-direct', 'read-staged', 'native-scatter', 'carry-active-kv', 'preserve-recurrent'):
            assert ident in deps[req]
        if event.startswith('read-'):
            assert read is None
            read = req, ident, event
            disk = hold = False
        elif event == 'disk-complete':
            assert read[:2] == (req, ident) and issued is None and not disk and sizes[ident][1] == 'SSD'
            disk = True
        elif event == 'disk-soft-hold':
            assert hold_ms and read[:2] == (req, ident) and issued is None and disk and not hold
            assert row['interval_us'] >= 1000 * hold_ms
            hold = True
        elif event == 'upload-submit':
            assert read[:2] == (req, ident) and issued is None and disk == (sizes[ident][1] == 'SSD')
            assert hold == bool(disk and hold_ms)
            assert row['ticket'] > last_fence
            issued = last_fence = row['ticket']
        elif event == 'upload-ready':
            assert read[:2] == (req, ident) and issued and row['ticket'] == issued
            observed.append(read)
            read = issued = None
    assert read is issued is None and native[-1]['charged_bytes'] == 0
    assert counts['disk-soft-hold'] == (counts['disk-complete'] if hold_ms else 0)
    assert order['native-target'] == order['compute-done'] == list(range(n))
    assert max(r['charged_bytes'] for r in native) == run['peak_prefetch_bytes']
    if run.get('qualification'):
        assert order['exact-state'] == list(range(n))
        logits = [i for i in range(n) if decisions[i] < sum(r['count'] for r in trace['requests'][i]['input_token_runs'])]
        assert order['exact-final-logits'] == logits
        assert all(r['identity'] == decisions[r['request']] and r['bytes'] > 0 for r in native
                   if r['event'] in ('exact-state', 'exact-final-logits'))
    controller_reads = [r['read'] for r in run['transfers']]
    assert [(r['request'], int(r['object_id'])) for r in controller_reads] == [(req, ident) for req, ident, _ in observed]
    assert all(r['lane_use'] == 'demand' for r, (_, _, event) in zip(controller_reads, observed, strict=True) if event == 'read-direct')
    return dict(scope='native event audit and optional correctness checks; no performance estimate',
                block=run['block'], policy=run['policy'], requests=n, boundary_counts=dict(Counter(decisions.values())),
                device_event_counts=dict(counts), capacity_bytes=run['capacity_bytes'], peak_prefetch_bytes=run['peak_prefetch_bytes'],
                direct_plus_scatter_requests=[i for i, events in by_request.items() if {'read-direct', 'native-scatter'} <= events],
                runtime_worker_sha256=run['worker_sha256'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'native', 'stderr', 'schedule', 'archive', 'out'):
        ap.add_argument('--' + name, required=True, type=Path)
    a = ap.parse_args()
    if a.archive.exists() or a.out.exists():
        raise FileExistsError('select new archive and summary paths')
    result = check(a.run, a.native, a.schedule)
    with ZipFile(a.archive, 'x') as z:
        for path in (a.run, a.native, a.stderr):
            z.writestr(ZipInfo(path.name), path.read_bytes(), compress_type=ZIP_DEFLATED)
    result['archive_sha256'] = digest(a.archive)
    with a.out.open('x', encoding='utf8', newline='\n') as out:
        print(json.dumps(result, indent=2, sort_keys=True), file=out)
    print(f"Cross-checked {result['requests']} requests and {result['device_event_counts']['upload-ready']} native transfers.")


if __name__ == '__main__':
    main()
