"""Validate greedy generation over the local pipe and report distinct latency clocks."""

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import json
from pathlib import Path
from statistics import mean
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from scripts.gpu.analysis.check_r10_catalog_fifo import check
from scripts.gpu.analysis.summarize_r10_pilot_aa import endpoints, p95
from scripts.validation.check_r10_native_catalog import digest


def distribution(xs):
    return dict(samples=len(xs), mean_ms=mean(xs), p95_ms=p95(xs), max_ms=max(xs)) if xs else dict(samples=0)


def first_visible_clock(events):
    """The terminal frame may flush the last incomplete UTF-8 sequence."""
    return next((Fraction(row['client_received_ms']) for row in events if row['text']), None)


def audit_sse(service, started, ttft_slo, tpot_slo):
    ttft, itl, tpot_rows, accept, total, ends = [], [], [], [], [], []
    satisfied = 0
    for req, (native, client) in enumerate(zip(service['requests'], service['http_sse_requests'], strict=True)):
        assert native['request'] == client['request'] == req
        events = client['events']
        assert [row['event'] for row in events] == ['token'] * len(native['tokens']) + ['end']
        start, accepted = started[req], Fraction(client['server_accept_ms'])
        clocks = [Fraction(r['client_received_ms']) for r in events]
        assert [start, accepted, *clocks] == sorted([start, accepted, *clocks])
        pieces = []
        for raw, wire, got in zip(native['tokens'], events[:-1], clocks[:-1], strict=True):
            assert all(raw[k] == wire[k] for k in ('id', 'native_us', 'index', 'piece_hex'))
            assert Fraction(raw['parent_observed_ms']) <= got
            pieces.append(bytes.fromhex(wire['piece_hex']))
        for key in ('reason', 'native_us'):
            assert native['finish'][key] == events[-1][key]
        assert Fraction(native['finish']['parent_observed_ms']) <= clocks[-1]
        assert b''.join(pieces).decode('utf8', errors='replace') == ''.join(r['text'] for r in events)
        accept.append(float(accepted - start))
        total.append(float(clocks[-1] - start))
        ends.append(clocks[-1])
        itl.extend(float(b - a) for a, b in zip(clocks[:-1], clocks[1:-1]))
        tpot = None
        if len(pieces) >= 2:
            tpot = float((clocks[-2] - clocks[0]) / (len(pieces) - 1))
            tpot_rows.append(tpot)
        visible = first_visible_clock(events)
        if visible is not None:
            first = float(visible - start)
            ttft.append(first)
            satisfied += ttft_slo is not None and first <= ttft_slo and (tpot is None or tpot <= tpot_slo)
    elapsed = float(max(ends) - started[0]) / 1000
    result = dict(scope='actual HTTP/SSE client timings on loopback; one frame per token; first visible decoded text for TTFT',
                  ttft_visible_ms=distribution(ttft), inter_token_frame_ms=distribution(itl), token_frame_tpot_ms=distribution(tpot_rows),
                  server_accept_delay_ms=distribution(accept), client_end_to_end_ms=distribution(total),
                  window_s=elapsed, completed=len(ends), emitted_tokens_per_s=sum(len(r['tokens']) for r in service['requests']) / elapsed,
                  completions_per_s=len(ends) / elapsed, requests_without_visible_output=len(ends) - len(ttft))
    if ttft_slo is not None:
        result.update(slo=dict(ttft_ms=ttft_slo, tpot_ms=tpot_slo, success_count=satisfied,
                               success_fraction=satisfied / len(ends), goodput_per_s=satisfied / elapsed))
    return result


def audit_failed_sse(receipt, expected, ttft_slo, tpot_slo):
    """Keep the full registered input in the denominator of an aborted arm."""
    if receipt.get('expected_service_requests', expected) != expected:
        raise ValueError('failed service run names a different request population')
    partial = receipt.get('partial_sse')
    if partial is None:
        return dict(scope='not enough client timing for a throughput estimate', expected=expected,
                    statuses=dict(unobserved=expected), slo_success_count=0, slo_success_fraction=0., goodput_per_s=None)
    rows = partial['requests']
    if [r['request'] for r in rows] != list(range(expected)):
        raise ValueError('partial SSE evidence omitted a registered request')
    statuses, successes = Counter(), 0
    firsts = []
    for r in rows:
        events, status = r['events'], r['status']
        kinds = [e['event'] for e in events]
        assert kinds == ['token'] * (len(events) - (status == 'terminal')) + (['end'] if status == 'terminal' else [])
        statuses[status] += 1
        if r['started_ms'] is not None:
            firsts.append(Fraction(r['started_ms']))
        if status != 'terminal':
            continue
        start = Fraction(r['started_ms'])
        clocks = [Fraction(e['client_received_ms']) for e in events]
        assert [start, Fraction(r['accepted_ms']), *clocks] == sorted([start, Fraction(r['accepted_ms']), *clocks])
        visible = first_visible_clock(events)
        tpot = (clocks[-2] - clocks[0]) / (len(clocks) - 2) if len(clocks) > 2 else None
        successes += visible is not None and visible - start <= ttft_slo and (tpot is None or tpot <= tpot_slo)
    end = partial['observation_end_ms']
    elapsed = float(Fraction(end) - min(firsts)) / 1000 if end is not None and firsts else 0.
    return dict(scope='partial client observation through aborted run; not a sustainable-throughput estimate',
                expected=expected, statuses=dict(statuses), slo_success_count=successes,
                slo_success_fraction=successes / expected, observation_s=elapsed,
                goodput_per_s=successes / elapsed if elapsed > 0 else None)


def audit(prefix, schedule, ttft_slo, tpot_slo):
    receipt_path, raw, _ = endpoints(prefix)
    receipt = json.loads(receipt_path.read_bytes())
    requests = receipt['service']['requests']
    dispatch = check(receipt_path, raw, schedule)
    n = dispatch['requests']
    assert len(requests) == n
    events = defaultdict(lambda: defaultdict(list))
    with raw.open(encoding='ascii', newline='') as f:
        for row in csv.DictReader(f):
            eq = row.pop('event')
            req = int(row.pop('request'))
            events[req][eq].append({k: int(v) for k, v in row.items()})
    started, completed = (list(map(Fraction, receipt[k])) for k in ('client_arrivals_ms', 'completions_ms'))
    delay, token_interval, per_token, prefill_to_first = [], [], [], []
    terminal, lengths, satisfied = Counter(), Counter(), 0
    for req, row in enumerate(requests):
        assert row['request'] == req
        tokens, finish = row['tokens'], row['finish']
        e = events[req]
        m = len(tokens)
        if receipt['qualification']:
            assert [v['identity'] for v in e['exact-service-tokens']] == [m]
        assert [v['identity'] for v in e['service-token']] == [token['index'] for token in tokens] == list(range(1, m + 1))
        assert [v['identity'] for v in e['service-decode-marker']] == list(range(1, m + int(finish['reason'] == 'eog')))
        native_start, = e['service-prefill-done']
        native_end, = e['compute-done']
        assert row['prefill_done']['native_us'] <= native_start['native_us'] <= finish['native_us'] <= native_end['native_us']
        assert all(a['native_us'] <= b['native_us'] <= finish['native_us'] for a, b in zip(e['service-token'], tokens, strict=True))
        token_at = [Fraction(t['parent_observed_ms']) for t in tokens]
        parent_times = [started[req], Fraction(row['prefill_done']['parent_observed_ms']), *token_at,
                        Fraction(finish['parent_observed_ms']), completed[req]]
        assert sorted(parent_times) == parent_times
        terminal[finish['reason']] += 1
        lengths[m] += 1
        if not m:
            continue
        first = float(token_at[0] - started[req])
        delay.append(first)
        prefill_to_first.append((tokens[0]['native_us'] - row['prefill_done']['native_us']) / 1000)
        tpot = float((token_at[-1] - token_at[0]) / (m - 1)) if m >= 2 else None
        token_interval.extend(float(b - a) for a, b in zip(token_at, token_at[1:]))
        if tpot is not None:
            per_token.append(tpot)
        satisfied += ttft_slo is not None and first <= ttft_slo and (tpot is None or tpot <= tpot_slo)
    seconds = float(completed[-1] - started[0]) / 1000
    out = dict(scope='complete one-slot process-pipe run; token ID frame receipt, not HTTP or production SLO',
               dispatch=dispatch, exact_recompute_qualification=receipt['qualification'], end_reason=terminal, output_lengths=lengths,
               client_pipe_ttft=distribution(delay), native_prefill_done_to_first_emission=distribution(prefill_to_first),
               client_pipe_inter_token_latency=distribution(token_interval), client_pipe_tpot=distribution(per_token),
               total_requests=n, requests_with_first_token=len(delay), elapsed_seconds=seconds,
               completion_per_s=n / seconds, emitted_tokens_per_s=sum(k * v for k, v in lengths.items()) / seconds)
    if ttft_slo is not None:
        out.update(slo=dict(ttft_ms=ttft_slo, tpot_ms=tpot_slo, one_token_tpot='not applicable; TTFT still required',
                            success_count=satisfied, success_fraction=satisfied / n, goodput_per_s=satisfied / seconds))
    if 'http_sse_requests' in receipt['service']:
        out['http_sse'] = audit_sse(receipt['service'], started, ttft_slo, tpot_slo)
    return out


def compare(first, second):
    receipts = [json.loads(endpoints(p)[0].read_bytes()) for p in (first, second)]
    a, b = receipts
    fields = ('block', 'capacity_bytes', 'worker_sha256', 'observer_sha256', 'seal_sha256', 'policy', 'qualification',
              'planned_arrival_offsets_ms', 'cost_profile_sha256')
    assert all(a[key] == b[key] for key in fields) and a['qualification'] and a['policy'] == 'forced'
    choices = [[d['action'] for d in sorted(row['decisions'], key=lambda x: x['snapshot']['request'])] for row in receipts]
    assert choices == [[0, 512, 512], [0, 1024, 512]]
    left, right = [r['service']['requests'] for r in receipts]
    for x, y in zip(left, right, strict=True):
        assert x['limit'] == y['limit'] and x['finish']['reason'] == y['finish']['reason']
        assert [t['id'] for t in x['tokens']] == [t['id'] for t in y['tokens']]
    return dict(scope='same-input forced shallow/deep exact generation qualification, no timing estimate',
                compared_requests=len(left), compared_output_tokens=sum(len(x['tokens']) for x in left), fixed_actions=choices)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for path in ('prefix', 'schedule', 'archive', 'out'):
        ap.add_argument('--' + path, type=Path, required=True)
    ap.add_argument('--ttft-slo-ms', type=float)
    ap.add_argument('--tpot-slo-ms', type=float)
    ap.add_argument('--paired-with', type=Path, help='check exact matching output from the deep arm of this qualification')
    a = ap.parse_args()
    if (a.ttft_slo_ms is None) != (a.tpot_slo_ms is None) or any(not 0 < v < float('inf') for v in (a.ttft_slo_ms, a.tpot_slo_ms) if v is not None):
        ap.error('supply both positive experimental SLOs, or neither')
    if a.archive.exists() or a.out.exists():
        raise FileExistsError('choose new evidence files')
    result = audit(a.prefix, a.schedule, a.ttft_slo_ms, a.tpot_slo_ms)
    inputs = [a.prefix]
    if a.paired_with:
        comparison = compare(a.prefix, a.paired_with)
        result = dict(comparison=comparison, shallow=result,
                      deep=audit(a.paired_with, a.schedule, a.ttft_slo_ms, a.tpot_slo_ms))
        inputs.append(a.paired_with)
    files = [p for prefix in inputs for p in endpoints(prefix)]
    if len({p.name for p in files}) != len(files):
        raise ValueError('prefix basenames must differ to preserve both transcripts')
    with ZipFile(a.archive, 'x') as z:
        for p in files:
            z.writestr(ZipInfo(p.name), p.read_bytes(), compress_type=ZIP_DEFLATED)
    with a.out.open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(dict(result, archive_sha256=digest(a.archive)), sort_keys=True, indent=2), file=f)


if __name__ == '__main__':
    main()
