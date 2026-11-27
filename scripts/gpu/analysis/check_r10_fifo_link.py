"""Reconstruct the continuous R10 diagnostic from both host and native ledgers."""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
from fractions import Fraction as F
from hashlib import sha256
from itertools import groupby
import json
from pathlib import Path

from restore_edge_select.live_fifo import Launch, LiveFIFO, Read
from restore_edge_select.live_snapshot import choose_observed
from scripts.gpu.experiments.run_r10_online_diagnostic import IDS, fixture, msjson


def serial(obj):
    return json.loads(json.dumps(obj, default=msjson))


class Replay:
    def __init__(self, effects):
        self.effects = iter(effects)

    def action(self, kind, now, **fields):
        expected = next(self.effects)
        actual = serial(dict(kind=kind, scheduler_ms=now, **fields))
        assert all(expected[k] == v for k, v in actual.items()), (expected, actual)
        assert F(expected['ack_ms']) >= now

    def reserve(self, request, deps, active_kv, now):
        self.action('reserve', now, request=request, deps=deps, active_kv=active_kv)

    def read(self, action, now):
        self.action('read', now, action=asdict(action))

    def compute(self, action, now):
        self.action('compute', now, action=asdict(action))
        return Launch(True)


def check_one(record, native):
    sizes = [r['bytes'] for r in native if r['event'] == 'catalog-size']
    capacity = record['capacity_bytes']
    assert all(r['capacity_bytes'] == capacity and 0 <= r['charged_bytes'] <= capacity for r in native)
    assert native[-1]['event'] == 'case-end' and native[-1]['charged_bytes'] == 0
    assert [r['event_no'] for r in native] == list(range(len(native)))
    assert [r['wall_us'] for r in native] == sorted(r['wall_us'] for r in native)
    events = defaultdict(dict)
    for i, (client, observed) in enumerate(zip(record['arrivals_client_ms'], record['arrivals_observed_ms'], strict=True)):
        t = F(observed)
        # Initial Windows Event.wait sometimes woke slightly early; preserve
        # the measured deviation for these archived diagnostics.
        assert F(client) >= record['arrivals_planned_ms'][i] - F(1, 10) and t >= F(client)
        events[t].setdefault('arrivals', []).append(i)
    for i, t in enumerate(record['completions_ms']):
        assert 'compute_done' not in events[F(t)]
        events[F(t)]['compute_done'] = i
    observed_reads = record['transfers']
    for line in observed_reads:
        t = F(line['end'])
        assert 'read_done' not in events[t]
        events[t]['read_done'] = Read(**line['read'])
    assert serial([F(end) - F(start) for end, start in zip(record['completions_ms'], record['arrivals_client_ms'], strict=True)]) == record['response_ms']

    chosen = iter(record['decisions'])
    force = record['force']

    def choose(exp, policy, snap):
        row = next(chosen)
        assert row['snapshot'] == serial(asdict(snap))
        selected = force[snap.request] if force is not None else choose_observed(exp, policy, snap)
        assert selected == row['action'] and row['choose_ns'] >= 0
        return selected

    replay = Replay(record['effects'])
    hold = record.get('controlled_ssd_hold_ms', 0)
    experiment = fixture(sizes, capacity, F(record['arrivals_planned_ms'][-1]), added_ssd_ms=hold)
    fifo = LiveFIFO(experiment, record['policy'], replay, choose=choose)
    for t, info in sorted(events.items()):
        fifo.step(t, **info)
    assert fifo.done and next(replay.effects, None) is next(chosen, None) is None
    assert fifo.chosen == [x['action'] for x in record['decisions']]
    assert fifo.peak_prefetch == record['peak_prefetch_bytes'] == max(r['charged_bytes'] for r in native)

    read_effects = [e for e in record['effects'] if e['kind'] == 'read']
    submits = [r for r in native if r['event'] in ('upload-submit', 'direct-upload-submit')]
    ready = [r for r in native if r['event'] in ('upload-ready', 'direct-upload-ready')]
    last = -1
    for e, s, r in zip(read_effects, submits, ready, strict=True):
        assert s['fence'] == r['fence'] == e['fence'] > 0
        assert s['object'] == r['object'] == IDS.index(e['action']['object_id'])
        assert s['bytes'] == r['bytes'] == sizes[s['object']]
        assert last < s['event_no'] < r['event_no']
        if hold and e['action']['object_id'].startswith('r'):
            assert F(e['native_read_call_ms']) >= F(e['hold_earliest_ms']) == F(e['ack_ms']) + hold
        else:
            assert 'hold_earliest_ms' not in e
        last = r['event_no']
    begin = [r for r in native if r['event'] == 'fifo-compute-submit']
    end = [r for r in native if r['event'] == 'fifo-compute-done']
    assert [r['request'] for r in begin] == [r['request'] for r in end] == [0, 1, 2]
    assert [r['bytes'] for r in begin] == fifo.chosen
    assert [r['bytes'] for r in end] == [0, 8, 8]
    assert all(r['status'] == 1 and r['fence'] > 0 for r in end)
    assert all(s['event_no'] < e['event_no'] for s, e in zip(begin, end, strict=True))
    assert all(e['event_no'] < s['event_no'] for e, s in zip(end[:-1], begin[1:], strict=True))
    assert len(fifo.ready_handoffs) == sum(r['event'] == 'active-kv-borrow' for r in native)
    if fifo.ready_handoffs:
        assert [r['bytes'] for r in native if r['event'] == 'active-kv-reused'] == [sizes[0]]
    return dict(commits=len(fifo.chosen), model_computes=3, verified_logits=16,
                staged_reads=sum(s['event'] == 'upload-submit' for s in submits),
                cold_reads=sum(s['event'] == 'direct-upload-submit' for s in submits),
                active_kv_handoffs=len(fifo.ready_handoffs), read_done_compute_done_ticks=sum(
                    'read_done' in e and 'compute_done' in e for e in events.values()))


def audit(prefix):
    raw, log = prefix.with_suffix('.csv'), prefix.with_suffix('.jsonl')
    cases = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()]
    stats = Counter()
    committed = set()
    with raw.open(encoding='utf-8', newline='') as f:
        for entry, (rep, rows) in zip(cases, groupby(csv.DictReader(f), lambda r: int(r['rep'])), strict=True):
            assert entry['rep'] == rep
            events = [{k: v if k in ('arm', 'event') else int(v) for k, v in r.items()} for r in rows]
            summary = check_one(entry, events)
            if rep >= 0:
                stats.update(summary)
                committed.add(tuple(d['action'] for d in entry['decisions']))
    drift = [float(F(actual) - planned) for row in cases[1:] for actual, planned in
             zip(row['arrivals_client_ms'], row['arrivals_planned_ms'], strict=True)]
    return dict(formal_blocks=len(cases) - 1, totals=stats, selected=sorted(committed), policy=cases[-1]['policy'],
                client_minus_planned_ms_range=[min(drift), max(drift)],
                controlled_ssd_hold_ms=cases[-1].get('controlled_ssd_hold_ms', 0),
                csv_sha256=sha256(raw.read_bytes()).hexdigest(), snapshots_sha256=sha256(log.read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('prefix', nargs='+', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    result = json.dumps(dict(scope='three-request catalogue, diagnostic timings include IPC, polling and exact readback',
                             data={p.name: audit(p) for p in a.prefix}), sort_keys=True, indent=2)
    if a.out:
        with a.out.open('x', encoding='utf-8') as f:
            print(result, file=f)
    print(result)


if __name__ == '__main__':
    main()
