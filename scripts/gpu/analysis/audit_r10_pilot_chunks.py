"""Audit an instrumented calibration block and localize long native intervals."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from scripts.gpu.analysis.check_r10_catalog_fifo import check
from scripts.gpu.analysis.summarize_r10_pilot_aa import endpoints
from scripts.validation.check_r10_native_catalog import digest


def union_ticks(intervals):
    last, total = 0, 0
    for a, b in sorted(intervals):
        total += max(0, b - max(a, last))
        last = max(b, last)
    return total


def audit(prefix, schedule):
    run_path, native_path, stderr = endpoints(prefix)
    run = json.loads(run_path.read_bytes())
    assert run['chunk_events'] and run['native_events'] and not run['qualification']
    base = check(run_path, native_path, schedule)
    requests = json.loads(schedule.read_bytes())['trace']['requests']
    actions = {row['snapshot']['request']: row['action'] for row in run['decisions']}
    by_request, completions = defaultdict(list), {}
    names = ('chunk-submit-begin', 'chunk-marker', 'chunk-ready')
    hz, warmup_grid_ms, calls, after_clock_line = None, None, [], 0
    for line in stderr.read_text(encoding='utf8', errors='replace').splitlines():
        if line.startswith('r10_chunk_clock,'):
            assert hz is None
            hz = int(line.split(',')[1])
            assert hz > 0
        elif line.startswith('r10_warmup_grid_us,'):
            assert warmup_grid_ms is None
            warmup_grid_ms = int(line.split(',')[1]) / 1000
            assert warmup_grid_ms > 0
        elif line.startswith('r10_vk_pipeline,'):
            a, b, tid, count, status = map(int, line.split(',')[1:])
            assert a <= b and count > 0 and status == 0
            calls.append((a, b, tid))
            after_clock_line += hz is not None
    assert bool(run.get('warmup_grid')) == (warmup_grid_ms is not None)
    if run.get('pipeline_events'):
        assert hz and calls, 'enabled pipeline witness did not report any calls'
    with native_path.open(encoding='ascii', newline='') as f:
        for row in csv.DictReader(f):
            event, i = row['event'], int(row['request'])
            if event in names:
                by_request[i].append((event, int(row['identity']), int(row['native_us']), int(row['interval_us']), int(row['ticket'])))
            elif event == 'compute-done':
                completions[i] = int(row['native_us'])
    samples = []
    for i, req in enumerate(requests):
        total = sum(x['count'] for x in req['input_token_runs'])
        ends = [min(x + 256, total) for x in range(actions[i], total, 256)]
        rows = by_request[i]
        assert len(rows) == 3 * len(ends)
        prior = None
        for j, end in enumerate(ends):
            a, b, c = rows[j * 3:j * 3 + 3]
            assert (a[0], b[0], c[0]) == names and a[1] == b[1] == c[1] == end
            begin, marked, ready = a[2], b[2], c[2]
            assert begin <= marked <= ready <= completions[i]
            assert prior is None or prior <= begin
            # The last attempted unsuccessful query was inside this window.
            assert 0 <= c[3] <= ready - marked
            sample = dict(request=i, boundary=actions[i], input_tokens=total, chunk=j + 1, end_tokens=end,
                          submit_to_marker_ms=(marked - begin) / 1000, marker_to_ready_ms=(ready - marked) / 1000,
                          since_last_unsuccessful_query_ms=c[3] / 1000 if c[3] else None,
                          previous_ready_to_submit_ms=(begin - prior) / 1000 if prior is not None else None)
            if hz:
                assert 0 < a[4] <= b[4] and 0 < b[3] <= marked - begin
                sample['host_llama_decode_ms'] = b[3] / 1000
            if run.get('pipeline_events'):
                overlap = [(max(a[4], x), min(b[4], y), tid) for x, y, tid in calls if a[4] < y and x < b[4]]
                sample.update(pipeline_calls_overlapping=len(overlap), pipeline_threads=len({tid for _, _, tid in overlap}),
                              pipeline_union_wall_ms=union_ticks((x, y) for x, y, _ in overlap) * 1000 / hz)
            samples.append(sample)
            prior = ready
    assert set(by_request) == set(range(len(requests)))
    measures = ('submit_to_marker_ms', 'marker_to_ready_ms', 'since_last_unsuccessful_query_ms', 'previous_ready_to_submit_ms')
    if hz:
        measures += ('host_llama_decode_ms',)
    if run.get('pipeline_events'):
        measures += ('pipeline_union_wall_ms',)
    stats = {}
    for field in measures:
        values = [r[field] for r in samples if r[field] is not None]
        stats[field] = dict(n=len(values), median=median(values) if values else None, max=max(values, default=None))
    return dict(scope='host intervals around submit and device timeline polling, not hardware compute durations',
                native_audit=base, stats=stats, chunks=len(samples),
                at_least_1000ms=[r for r in samples if any((r[k] or 0) >= 1000 for k in measures)],
                pipeline_calls_total=len(calls), pipeline_calls_logged_after_chunk_clock=after_clock_line,
                qpc_frequency_hz=hz, warmup_grid_host_ms=warmup_grid_ms, worker_sha256=run['worker_sha256'],
                observer_sha256=run.get('observer_sha256'), schedule_sha256=digest(schedule))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('prefix', 'schedule', 'archive', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    args = ap.parse_args()
    if args.archive.exists() or args.out.exists():
        raise FileExistsError('select new evidence filenames')
    result = audit(args.prefix, args.schedule)
    with ZipFile(args.archive, 'x') as archive:
        for path in endpoints(args.prefix):
            archive.writestr(ZipInfo(path.name), path.read_bytes(), compress_type=ZIP_DEFLATED)
    result['archive_sha256'] = digest(args.archive)
    with args.out.open('x', encoding='utf8', newline='\n') as out:
        print(json.dumps(result, indent=2, sort_keys=True), file=out)
    print(f"Audited {result['chunks']} chunks over {result['native_audit']['requests']} requests.")


if __name__ == '__main__':
    main()
