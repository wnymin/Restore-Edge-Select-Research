"""Exact joint-domain audit: active compute residual and future SSD multiplier.

Research evaluator, not a low-overhead scheduler. Rational half-plane splits
retain open faces, equality edges, and vertices, including event-priority jumps.
The existing in-flight transfer, if any, has a fixed completion time.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
import csv
import json
from pathlib import Path

from scripts.cpu.lookahead.robustness.study_lookahead_first_decision import Capture
from scripts.cpu.lookahead.screening.study_lookahead_phase1 import Experiment, ROOT, read
from scripts.cpu.lookahead.screening.study_lookahead_workload_bounds import clone, random_cases


@dataclass(frozen=True)
class Linear:
    a: F = F(0)
    r: F = F(0)
    x: F = F(0)

    def at(self, point):
        return self.a + self.r * point[0] + self.x * point[1]

    def __add__(self, other):
        if not isinstance(other, Linear):
            other = Linear(F(other))
        return Linear(self.a + other.a, self.r + other.r, self.x + other.x)

    __radd__ = __add__

    def __neg__(self):
        return Linear(-self.a, -self.r, -self.x)

    def __sub__(self, other):
        if not isinstance(other, Linear):
            other = Linear(F(other))
        return self + (-other)


@dataclass(frozen=True)
class Region:
    # Ordered vertices of the closure; also supports segments and singletons.
    vertices: tuple
    strict: tuple = ()

    @classmethod
    def rectangle(cls, rlo, rhi, xlo, xhi):
        assert 0 < rlo < rhi and 0 < xlo < xhi
        return cls(((rlo, xlo), (rhi, xlo), (rhi, xhi), (rlo, xhi)))

    def center(self):
        n = len(self.vertices)
        return tuple(sum(p[i] for p in self.vertices) / n for i in range(2))

    def contains(self, point):
        # Used only for points known to lie in the closure.
        return all(d.at(point) < 0 for d in self.strict)

    @lru_cache(maxsize=100000)
    def clip(self, d, strict=False):
        """Intersect with d <= 0 (or d < 0), without dropping boundary strata."""
        out = []
        prev = self.vertices[-1]
        vp = d.at(prev)
        for curr in self.vertices:
            vc = d.at(curr)
            if (vp <= 0) != (vc <= 0):
                t = vp / (vp - vc)
                out.append(tuple(a + t * (b - a) for a, b in zip(prev, curr)))
            if vc <= 0:
                out.append(curr)
            prev, vp = curr, vc
        points = []
        for point in out:
            if point not in points:
                points.append(point)
        if not points:
            return None
        result = Region(tuple(points), self.strict + ((d,) if strict else ()))
        # The vertex barycenter is in the relative interior of the closure.
        # A strict constraint excluding that point excludes the entire face.
        return result if result.contains(result.center()) else None

    def dimension(self):
        if len(self.vertices) == 1:
            return 0
        a, b = self.vertices[:2]
        return 2 if any((b[0]-a[0])*(p[1]-a[1]) !=
                        (b[1]-a[1])*(p[0]-a[0]) for p in self.vertices[2:]) else 1


class Split(Exception):
    def __init__(self, parts):
        self.parts = parts


def sign(region, d):
    if not d.r and not d.x:
        return (d.a > 0) - (d.a < 0)
    values = [d.at(p) for p in region.vertices]
    if min(values) > 0:
        return 1
    if max(values) < 0:
        return -1
    if min(values) == max(values) == 0:
        return 0
    parts = []
    negative = region.clip(d, True)
    if negative is not None:
        parts.append((-1, negative))
    zero = region.clip(d)
    if zero is not None:
        zero = zero.clip(-d)
        if zero is not None:
            parts.append((0, zero))
    positive = region.clip(-d, True)
    if positive is not None:
        parts.append((1, positive))
    assert parts
    if len(parts) > 1:
        raise Split([part for _, part in parts])
    return parts[0][0]


def symbolic(exp, state, plan, estimate, speeds, region, parameter='future-ssd'):
    now, head, n, admitted, ready, running, transfer = clone(state)[:7]
    assert running is not None and running > now
    assert max(exp.arrivals[:n]) <= now
    assert not state[8] and state[9] is None and not state[10]
    assert len(plan) == n
    assert parameter in ('future-ssd','active-pair')
    assert parameter!='active-pair' or transfer is not None
    running = Linear(now, F(1))
    now = Linear(now)
    if transfer is not None:
        transfer = ((Linear(state[0],0,1) if parameter=='active-pair' else Linear(transfer[0])), transfer[1])
    active = state[7]
    resume = True
    ends, events = [], []

    def compare(a, b, kind):
        s = sign(region, a-b)
        return s == 0 if kind == 'eq' else s < 0

    def occupied():
        return set().union(*admitted.values()) - exp.initial

    while head < n:
        if not resume and transfer is not None and compare(transfer[0], now, 'eq'):
            ready.add(transfer[1])
            events.append(('ready', head, transfer[1]))
            transfer = None
        if not resume and running is not None and compare(running, now, 'eq'):
            ends.append(now)
            events.append(('finish', head))
            head += 1
            running = None
            if head == n:
                break
        deps = exp.candidates[head][plan[head]]
        if running is None and ready.issuperset(deps):
            admitted.pop(head, None)
            shared = bool(deps) and (deps[-1] in exp.initial or deps[-1] in occupied())
            copy = (exp.path_time('device_copy', exp.objects[deps[-1]]['bytes']) *
                    estimate.get('device_copy', 1)
                    if shared and plan[head] < exp.lengths[head] else 0)
            active = deps
            running = now + copy + exp._suffix_times[head][plan[head]] * estimate.get('compute', 1)
            ready.intersection_update(occupied() | exp.initial)
            events.append(('start', head, shared))
            if compare(running, now, 'eq'):
                resume = False
                continue
        for j in range(head+1, min(n, head+1+exp.window)):
            future = set(exp.candidates[j][plan[j]]) - exp.initial
            if j not in admitted and future and exp.size(occupied() | future) <= exp.budget:
                admitted[j] = future
                if running is not None:
                    ready.update(future.intersection(active[:-1]) - ready)
                events.append(('admit', head, j))
        if transfer is None:
            order = [] if running is not None else list(deps)
            for j in sorted(admitted):
                order.extend(exp.candidates[j][plan[j]])
            for obj in order:
                if obj not in ready:
                    path = {'RAM': 'ram_to_device', 'SSD': 'ssd_to_device'}[exp.objects[obj]['tier']]
                    duration = exp._transfer_times[obj] * estimate.get(path, 1) / speeds[path]
                    if path == 'ssd_to_device' and parameter=='future-ssd':
                        duration = Linear(0, 0, duration)
                    transfer = (now + duration, obj)
                    events.append(('transfer', head, obj))
                    break
        times = [t for t in (running, transfer[0] if transfer else None) if t is not None]
        assert times
        now = times[0]
        for t in times[1:]:
            if compare(t, now, 'lt'):
                now = t
        resume = False
        assert len(events) < 10000, 'research event guard reached'
    assert all(t.r in (0, 1) and t.x >= 0 for t in ends)
    return ends, tuple(events)


def partition(exp, state, plans, estimate, speeds, rectangle, parameter='future-ssd'):
    """Common event refinement for all candidates; disjoint sign strata."""
    pending = [rectangle]
    cells = []
    while pending:
        region = pending.pop()
        try:
            vectors = {c: symbolic(exp, state, p, estimate, speeds, region,parameter)[0]
                       for c, p in plans.items()}
        except Split as split:
            pending.extend(split.parts)
        else:
            cells.append((region, vectors))
        assert len(cells) + len(pending) < 5000, 'research cell guard reached'
    return cells


def oracle(exp, state, plan, estimate, speeds, point, parameter='future-ssd'):
    changed = (*state[:5], state[0] + point[0], *state[6:])
    if parameter=='active-pair':
        assert state[6] is not None
        changed=(*changed[:6],(state[0]+point[1],state[6][1]),*changed[7:])
        costs=estimate
    else:
        assert parameter=='future-ssd'
        costs = estimate | {'ssd_to_device': estimate.get('ssd_to_device', 1) * point[1]}
    return Experiment._run(exp, 'fixed', 0, plan, 100, state[2], snapshot=clone(changed),
                           estimate=costs, forecast_speed=speeds, lean_rollout=True)['completion_ms']


def analyze(exp, record, radius=F(1,4), rectangle=None, parameter='future-ssd'):
    target, fallback, state, base, estimate, speeds = record
    residual = state[5] - state[0]
    assert parameter!='active-pair' or rectangle is not None
    rectangle = rectangle or Region.rectangle(residual*(1-radius), residual*(1+radius),
                                               1-radius, 1+radius)
    plans = {c: tuple(c if j == target else b for j, b in enumerate(base))
             for c in exp.eligible[target]}
    cells = partition(exp, state, plans, estimate, speeds, rectangle,parameter)
    upper = {c: None for c in plans}
    total_upper = {c: None for c in plans}
    regret = {c: F(0) for c in plans}
    attained = {c: None for c in plans}
    probes = 0
    dimensions = Counter()
    sensitive = [False, False]
    for region, vectors in cells:
        dimensions[region.dimension()] += 1
        totals = {c: sum(v, Linear()) for c, v in vectors.items()}
        for c in plans:
            diffs = [a-b for a, b in zip(vectors[c], vectors[fallback])]
            sensitive[0] |= any(d.r != 0 for d in diffs)
            sensitive[1] |= any(d.x != 0 for d in diffs)
            h = max(d.at(p) for d in diffs for p in region.vertices)
            t = max((totals[c]-totals[fallback]).at(p) for p in region.vertices)
            upper[c] = h if upper[c] is None else max(upper[c], h)
            total_upper[c] = t if total_upper[c] is None else max(total_upper[c], t)
            regret[c] = max(regret[c], *(max((totals[c]-totals[b]).at(p)
                                             for p in region.vertices) for b in plans))
        # Only evaluate a branch formula at actual points belonging to it.
        # Excluded vertices represent one-sided limits, not endpoint schedules.
        center = region.center()
        points = {center} | {p for p in region.vertices if region.contains(p)}
        points |= {tuple((a+b)/2 for a, b in zip(center, p)) for p in region.vertices}
        for point in points:
            actual = {c: oracle(exp, state, plan, estimate, speeds, point,parameter)
                      for c, plan in plans.items()}
            for c, vector in vectors.items():
                assert actual[c] == [t.at(point) for t in vector], (target, c, point)
                h = max(a-b for a, b in zip(actual[c], actual[fallback]))
                if attained[c] is None or h > attained[c][0]:
                    attained[c] = (h, point)
            probes += len(plans)
    corner_upper = {c: None for c in plans}
    for point in rectangle.vertices:
        actual = {c: oracle(exp, state, p, estimate, speeds, point,parameter) for c, p in plans.items()}
        for c in plans:
            h = max(a-b for a, b in zip(actual[c], actual[fallback]))
            corner_upper[c] = h if corner_upper[c] is None else max(corner_upper[c], h)
        probes += len(plans)
    safe = [c for c in plans if upper[c] <= 0]
    assert fallback in safe
    choose = lambda allowed: min(allowed, key=lambda c: (regret[c], c != fallback, -c))
    robust, protected = choose(plans), choose(safe)
    rows = [dict(target=target, head=state[1], fallback=fallback, candidate=c,
                 cells=len(cells), faces=dimensions[2], edges=dimensions[1], vertices=dimensions[0],
                 probes=probes, rlo=str(min(p[0] for p in rectangle.vertices)),
                 rhi=str(max(p[0] for p in rectangle.vertices)),
                 xlo=str(min(p[1] for p in rectangle.vertices)),
                 xhi=str(max(p[1] for p in rectangle.vertices)),
                 inflight=int(state[6] is not None), sensitive_r=int(sensitive[0]),
                 sensitive_x=int(sensitive[1]), individual_upper=str(upper[c]),
                 total_upper=str(total_upper[c]), regret=str(regret[c]),
                 corner_upper=str(corner_upper[c]), safe=int(c in safe),
                 robust=robust, protected=protected,
                 corners_false_safe=int(corner_upper[c] <= 0 < upper[c]),
                 sampled_max=str(attained[c][0]),
                 sampled_r=str(attained[c][1][0]), sampled_x=str(attained[c][1][1]))
            for c in plans]
    return rows


def geometry_checks():
    box = Region.rectangle(F(1), F(3), F(1), F(3))
    d = Linear(0, 1, -1)
    parts = [box.clip(d, True), box.clip(d).clip(-d), box.clip(-d, True)]
    assert [p.dimension() for p in parts] == [2, 1, 2]
    assert parts[1].vertices == ((F(1), F(1)), (F(3), F(3)))
    assert parts[1].clip(d, True) is None
    point = parts[1].clip(Linear(-1, 1))
    assert point.dimension() == 0 and point.center() == (F(1), F(1))
    assert point.clip(Linear(1, -1), True) is None
    # An excluded boundary can still determine the supremum.
    half = box.clip(Linear(-2, 1), True)
    assert max(p[0] for p in half.vertices) == 2 and not half.contains((F(2), F(2)))


def constructed_check():
    payload = read(ROOT/'docs/evidence/dynamic-lookahead-snapshot-gap-constructed-witness.json')
    model = read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    exp = Experiment(payload['trace'], model, 256)
    state = (F(0), 0, 2, {}, set(exp.initial), F(30), None, (), set(), None, {})
    record = (1, 0, state, (0, 0), {}, {'ram_to_device': F(1), 'ssd_to_device': F(1)})
    rows = analyze(exp, record, rectangle=Region.rectangle(F(30), F(75,2), F(3,4), F(5,4)))
    deep = next(r for r in rows if r['candidate'] == 1152)
    assert deep['individual_upper'] == '1' and not deep['safe']
    assert deep['protected'] == 0
    return rows


def run(args):
    geometry_checks()
    constructed_check()
    model = read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    rows, stats = [], Counter()
    for case, (trace, labels, estimate) in enumerate(random_cases(model, args.seed, args.rounds)):
        if not args.native_arrivals:
            for request in trace['requests']:
                request['arrival_ms'] = 0
        exp = Capture(trace, model, labels['budget'])
        exp.run_trigger(estimate, 'full')
        stats['configurations'] += 1
        for record in exp.records:
            state = record[2]
            if state[5] is None or state[5] <= state[0]:
                stats['skipped_no_positive_compute'] += 1
                continue
            result = analyze(exp, record, F(args.radius))
            stats['decisions'] += 1
            stats['candidates'] += len(result)
            first = result[0]
            for name in ('cells', 'faces', 'edges', 'vertices', 'probes', 'inflight',
                         'sensitive_r', 'sensitive_x'):
                stats[name] += first[name]
            stats['both_sensitive'] += first['sensitive_r'] and first['sensitive_x']
            stats['constraint_changes_action'] += first['protected'] != first['robust']
            stats['corners_false_safe'] += sum(r['corners_false_safe'] for r in result)
            rows.extend(dict(seed=args.seed, case=case, decision=stats['decisions'], **r) for r in result)
            if args.witness and not args.witness.exists() and any(r['corners_false_safe'] for r in result):
                payload = dict(seed=args.seed, case=case, trace=trace, budget=labels['budget'],
                               target=record[0], fallback=record[1], base=record[3],
                               estimate=record[4], speeds=record[5], state=record[2], rows=result)
                with args.witness.open('x', encoding='utf-8') as f:
                    json.dump(payload, f, default=lambda v: str(v) if isinstance(v,F) else sorted(v),
                              ensure_ascii=False, indent=2)
        if stats['decisions'] >= args.limit:
            break
    assert rows
    with args.out.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with args.summary.open('x', encoding='utf-8') as f:
        json.dump(dict(stats), f, sort_keys=True, indent=2)
    print(json.dumps(dict(stats), sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=7770)
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--limit', type=int, default=300)
    parser.add_argument('--radius', default='1/4')
    parser.add_argument('--native-arrivals', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--witness', type=Path)
    args = parser.parse_args()
    assert 0 < F(args.radius) < 1
    run(args)
