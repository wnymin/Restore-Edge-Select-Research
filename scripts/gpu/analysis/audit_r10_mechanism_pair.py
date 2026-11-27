"""Check shallow/deep mechanism pairs against actual native actions and timing."""

import argparse
from collections import Counter
import csv
from fractions import Fraction
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from scripts.gpu.analysis.check_r10_catalog_fifo import check
from scripts.gpu.analysis.summarize_r10_pilot_aa import endpoints
from scripts.validation.check_r10_native_catalog import digest


def phases(events):
    finishes = {e['request']: e['relative_head_marker_ms'] for e in events if e['event'] == 'compute-done'}
    head_done = finishes[0]
    result = {}
    for i in finishes:
        own = [e for e in events if e['request'] == i]
        one = {e['event']: e['relative_head_marker_ms'] for e in own if e['event'] in ('native-target', 'suffix-submitted', 'reserved')}
        reads = Counter(e['event'] for e in own if e['event'] in ('read-direct', 'read-staged'))
        copies = Counter()
        for e in own:
            if e['event'] in ('native-scatter', 'preserve-old-active', 'preserve-recurrent'):
                copies[e['event']] += e['interval_us']
        result[i] = dict(request=i, relative_head_marker_target_ms=one['native-target'],
                         relative_head_marker_suffix_ms=one['suffix-submitted'], native_finish_ms=finishes[i],
                         target_to_suffix_ms=one['suffix-submitted'] - one['native-target'],
                         suffix_to_finish_ms=finishes[i] - one['suffix-submitted'],
                         reserved_before_head_done=one.get('reserved', float('inf')) <= head_done,
                         finished_uploads_before_head_done=sum(
                             e['event'] == 'upload-ready' and e['relative_head_marker_ms'] <= head_done for e in own),
                         bytes_uploaded=sum(e['bytes'] for e in own if e['event'] == 'upload-ready'),
                         soft_disk_hold_ms=sum(e['interval_us'] for e in own if e['event'] == 'disk-soft-hold') / 1000,
                         reads=dict(reads), device_copy_ms={k: v / 1000 for k, v in copies.items()})
    return list(result.values())


def arm(name, prefix, schedule, control):
    receipt, native, stderr = endpoints(prefix)
    run = json.loads(receipt.read_bytes())
    assert run['policy'] == 'forced' and run['forced_actions_sha256'] == digest(control)
    anchor = run['publish_rest_on']
    assert anchor in ('first-compute', 'first-ssd') and run['native_events'] and run.get('warmup_grid')
    shape = check(receipt, native, schedule)
    assert shape['requests'] == 3
    expected = json.loads(control.read_bytes())
    assert expected['schedule_sha256'] == digest(schedule)
    decisions = sorted(run['decisions'], key=lambda x: x['snapshot']['request'])
    actions = [x['action'] for x in decisions]
    assert actions == expected['boundaries'] and actions[0] in (0, 512)
    assert actions[1:] == [{'shallow': 512, 'deep': 1024}[name], 512]
    s = decisions[1]['snapshot']
    assert s['head'] == 0 and s['arrived'] == 3 and s['committed'] == [actions[0]]
    if anchor == 'first-compute':
        assert s['transfer_object'] is None and s['suffix_start_ms'] is not None
        time = s['suffix_start_ms']
    else:
        trace = json.loads(schedule.read_bytes())['trace']
        assert actions[0] == 512 and s['suffix_start_ms'] is None and s['transfer_start_ms'] is not None
        assert any(str(o['id']) == s['transfer_object'] and o['tier'] == 'SSD' for o in trace['objects'])
        time = s['transfer_start_ms']
    state = {k: v for k, v in s.items() if not k.endswith('_ms')}
    state['ready'] = sorted(state['ready'])
    age = float(Fraction(s['now_ms']) - Fraction(time))
    arrival = list(map(Fraction, run['client_arrivals_ms']))
    completion = list(map(Fraction, run['completions_ms']))
    assert arrival[0] <= arrival[1] == arrival[2]
    events = []
    with native.open(encoding='ascii', newline='') as f:
        for row in csv.DictReader(f):
            event = row.pop('event')
            events.append(dict(event=event, **{k: int(v) for k, v in row.items()}))
    if anchor == 'first-compute':
        head = next(x['native_us'] for x in events if x['request'] == 0 and x['event'] == 'suffix-submitted')
    else:
        head = next(x['native_us'] for x in events if x['request'] == 0 and
                    x['event'] in ('read-staged', 'read-direct') and str(x['identity']) == s['transfer_object'])
    for x in events:
        x['relative_head_marker_ms'] = (x.pop('native_us') - head) / 1000
    diagnostic = stderr.read_text(encoding='utf8', errors='replace').splitlines()
    after_start = False
    pipeline_calls_in_slot = 0
    for line in diagnostic:
        if line.startswith('r10_chunk_clock,'):
            after_start = True
        if after_start and line.startswith('r10_vk_pipeline,'):
            pipeline_calls_in_slot += 1
    stats = dict(audit=shape, actions=actions, snapshot_shape=state, controller_head_elapsed_ms=age,
                 response_ms=[float(b - a) for a, b in zip(arrival, completion, strict=True)],
                 native_finish_from_head_marker_ms=[x['relative_head_marker_ms'] for x in events if x['event'] == 'compute-done'],
                 phases=phases(events), observed_pipeline_calls_logged_after_start=pipeline_calls_in_slot,
                 event_graph=[x for x in events if not x['event'].startswith(('chunk-', 'exact-'))],
                 prefill_warmup_us=[int(line.split(',')[1]) for line in diagnostic if line.startswith('r10_warmup_grid_us,')])
    assert len(stats['native_finish_from_head_marker_ms']) == 3 and len(stats['prefill_warmup_us']) == 1
    return run, stats


def pair(shallow, deep, schedule, shallow_control, deep_control):
    ra, a = arm('shallow', shallow, schedule, shallow_control)
    rb, b = arm('deep', deep, schedule, deep_control)
    fixed = ('capacity_bytes', 'seal_sha256', 'worker_sha256', 'observer_sha256', 'publish_rest_on',
             'qualification', 'chunk_events', 'pipeline_events', 'resources', 'cost_profile_sha256')
    assert {k: ra.get(k) for k in fixed} == {k: rb.get(k) for k in fixed}
    assert ra.get('ssd_hold_ms', 0) == rb.get('ssd_hold_ms', 0)
    assert a['snapshot_shape'] == b['snapshot_shape']
    responses = [x - y for x, y in zip(a['response_ms'], b['response_ms'], strict=True)]
    return dict(scope='one same-input anchored pair; positive favors deeper, reporting only observed actions',
                capacity_bytes=ra['capacity_bytes'], schedule_sha256=digest(schedule), shallow=a, deep=b,
                publication_anchor=ra['publish_rest_on'],
                soft_ssd_hold_ms=ra.get('ssd_hold_ms', 0),
                controller_anchor_elapsed_difference_ms=abs(a['controller_head_elapsed_ms'] - b['controller_head_elapsed_ms']),
                shallow_minus_deep_response_ms=responses, shallow_minus_deep_total_response_ms=sum(responses))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('shallow', 'deep', 'schedule', 'shallow-control', 'deep-control', 'archive', 'out'):
        ap.add_argument('--' + key, type=Path, required=True)
    args = ap.parse_args()
    if args.archive.exists() or args.out.exists():
        raise FileExistsError('select new evidence filenames')
    r = pair(args.shallow, args.deep, args.schedule, args.shallow_control, args.deep_control)
    with ZipFile(args.archive, 'x') as out:
        for path in (*endpoints(args.shallow), *endpoints(args.deep), args.shallow_control, args.deep_control):
            out.writestr(ZipInfo(path.name), path.read_bytes(), compress_type=ZIP_DEFLATED)
    with args.out.open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(dict(r, archive_sha256=digest(args.archive)), sort_keys=True, indent=2), file=f)


if __name__ == '__main__':
    main()
