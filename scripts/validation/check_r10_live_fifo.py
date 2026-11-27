"""Check dispatch, timestamp ties and chooser observations against R07 execution."""

import argparse
from collections import Counter
from dataclasses import replace
from fractions import Fraction as F
import json
from pathlib import Path

from restore_edge_select.live_fifo import Launch, LiveFIFO
from restore_edge_select.live_snapshot import audit_simulated_decisions
from restore_edge_select.simulator import Experiment
from scripts.cpu.replay.study_r07_arrivals import CONFIG, cases, read
from scripts.gpu.experiments.run_r10_online_diagnostic import fixture, MIB, POLICIES


class VirtualDevice:
    """A deterministic driver for testing event equivalence, not a performance model."""

    def __init__(self, exp):
        self.exp = exp
        self.read_end = self.computed = self.suffix = None
        self.status = Counter()

    def reserve(self, request, deps, active_kv, now):
        self.status['reserves'] += 1
        self.status['active_kv_handoffs'] += bool(active_kv)

    def read(self, action, now):
        assert self.read_end is None
        path = 'ram_to_device' if self.exp.objects[action.object_id]['tier'] == 'RAM' else 'ssd_to_device'
        duration = self.exp._actual_transfer_times[action.object_id] / self.exp.io_epochs[path][1][0]
        self.read_end = now + duration
        self.status[action.lane_use] += 1

    def compute(self, action, now):
        assert self.computed is None and self.suffix is None
        copy = (self.exp.path_time('device_copy', self.exp.objects[action.deps[-1]]['bytes'])
                if action.shared_checkpoint else 0)
        end = now + copy + self.exp.recompute(action.request, action.boundary)
        if end != now:
            self.computed = end
        if copy:
            self.suffix = now + copy
            self.status['shared_checkpoint_copy'] += 1
        self.status['zero_compute'] += end == now
        return Launch(not copy, end == now)

    def drive(self, fifo):
        while not fifo.done:
            t = min(x for x in (self.read_end, self.computed, self.suffix,
                                self.exp.arrivals[fifo.arrived] if fifo.arrived < len(self.exp.requests) else None)
                    if x is not None)
            arrived = []
            for i in range(fifo.arrived, len(self.exp.requests)):
                if self.exp.arrivals[i] != t:
                    break
                arrived.append(i)
            rd = fifo.reading if t == self.read_end else None
            start = fifo.head if t == self.suffix else None
            done = fifo.head if t == self.computed else None
            if rd:
                self.read_end = None
            if start is not None:
                self.suffix = None
            if done is not None:
                self.computed = None
            if rd and done is not None and arrived:
                self.status['arrival_read_compute_tie'] += 1
            fifo.step(t, arrivals=arrived, read_done=rd, suffix_started=start, compute_done=done)


def compare(exp, policy='local-cost', forced=None):
    original_decisions, decisions = [], []

    def capture(where):
        def add(snap, action):
            where.append((replace(snap, admitted=tuple((i, tuple(sorted(deps))) for i, deps in snap.admitted)), action))
        return add

    expected = (audit_simulated_decisions(exp, policy, capture(original_decisions)) if forced is None else
                exp.run('fixed', boundaries=forced))
    driver = VirtualDevice(exp)
    fifo = LiveFIFO(exp, policy, driver, observe=capture(decisions),
                    choose=None if forced is None else lambda model, p, snap: forced[snap.request])
    driver.drive(fifo)
    assert fifo.chosen == expected['boundaries_tokens']
    assert fifo.completions == expected['completion_ms']
    assert fifo.peak_prefetch == expected['peak_prefetch_bytes']
    assert [fifo.reservations.get(i) for i in range(fifo.head)] == expected['prefetch_reserved_at_ms']
    timeline = [(x.object_id, exp.requests[x.request]['id'], x.lane_use, start, end) for x, start, end in fifo.transfers]
    expected_transfer = [r for r in expected['transfers'] if r['path'] != 'device_copy']
    assert timeline == [(r['object'], r['request'], r['mode'], r['start_ms'], r['end_ms']) for r in expected_transfer]
    assert sum(command.shared_checkpoint for command, _ in fifo.starts) == len(expected['transfers']) - len(expected_transfer)
    if forced is None:
        assert decisions == original_decisions, next(zip(decisions, original_decisions))
    driver.status['committed'] += len(decisions)
    driver.status['transfer_events'] += len(timeline)
    return fifo, driver.status


def directed():
    sizes = [16_777_216, 52_690_944, 16_777_216, 52_690_944]
    standard = fixture(sizes, sum(sizes), F(256))
    spec = dict(context_tokens=2048, payload=dict(checkpoint_bytes=sizes[1], attention_bytes_per_token=sizes[0] // 512))

    def new(t, cap, *, tied=False, empty_suffix=False):
        requests = [dict(r) for r in standard.trace['requests']]
        requests[2]['arrival_ms'] = str(t)
        if empty_suffix:
            requests[1]['input_token_runs'] = requests[1]['input_token_runs'][:-1]
        trace = dict(standard.trace, requests=requests)
        if tied:
            trace['cost'] = dict(trace['cost'], ram_to_device=dict(startup_ms=0, bytes_per_ms=sizes[0]),
                                 ssd_to_device=dict(startup_ms=0, bytes_per_ms=str(F(sizes[1], 255))))
        return Experiment(trace, spec, F(cap, MIB))

    tie, stats = compare(new(256, sum(sizes), tied=True), forced=(0, 512, 1024))
    assert stats['arrival_read_compute_tie'] == 1 and stats['active_kv_handoffs'] == 1
    assert tie.completions[0] == tie.reservations[2] == 256
    shared, link = compare(new(10, sum(sizes)), forced=(0, 512, 512))
    assert link['shared_checkpoint_copy'] == 1
    cold, plain = compare(new(10, sum(sizes[:2]) - 65536), forced=(0, 512, 1024))
    assert cold.peak_prefetch == 0 and plain['demand'] == 5
    _, empty = compare(new(256, sum(sizes), tied=True, empty_suffix=True), forced=(0, 512, 1024))
    assert empty['zero_compute'] == 1
    return {'three_way_tie': dict(stats), 'shared_checkpoint': dict(link), 'cold_head': dict(plain), 'zero_suffix': dict(empty)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    counts = {p: Counter() for p in POLICIES}
    for key, exp in cases(read(CONFIG), 'short'):
        for p in POLICIES:
            try:
                _, totals = compare(exp, p)
            except Exception as exc:
                raise RuntimeError((key, p)) from exc
            counts[p].update(totals)
            counts[p]['blocks'] += 1
    assert all(row['blocks'] == 60 and row['committed'] == 420 for row in counts.values())
    result = dict(scope='CPU deterministic execution equivalence, before physical attachment', original_cases=counts,
                  directed=directed())
    output = json.dumps(result, sort_keys=True, indent=2)
    if args.out:
        with args.out.open('x', encoding='utf-8') as f:
            print(output, file=f)
    print(output)


if __name__ == '__main__':
    main()
