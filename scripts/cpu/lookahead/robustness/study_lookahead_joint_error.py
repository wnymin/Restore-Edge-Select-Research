"""Audit common-error bounds and FIFO block evaluation without changing the simulator."""

import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
from dataclasses import replace
from fractions import Fraction
from itertools import accumulate, product
from pathlib import Path
from time import perf_counter_ns

from scripts.cpu.lookahead.screening.study_lookahead_bounds import Case, fixed, path, sample, selective


def hull(*envelopes):
    """Upper envelope of a + b*x, returning active lines and their breakpoints."""
    lines = {}
    for es in envelopes:
        for a, b in es:
            lines[b] = max(a, lines.get(b, a))
    active, knots = [], []
    for b, a in sorted(lines.items()):
        while active:
            pa, pb = active[-1]
            cross = Fraction(pa - a, b - pb)
            if not knots or cross > knots[-1]:
                break
            active.pop()
            knots.pop()
        if active:
            knots.append(cross)
        active.append((a, b))
    return tuple(active), tuple(knots)


def shift(envelope, a, b=0):
    lines, knots = envelope
    return tuple((p + a, q + b) for p, q in lines), knots


def affine_path(case, deep):
    const = lambda a: (((a * 100, 0),), ())
    maximum = lambda u, v: hull(u[0], v[0])
    lane = const(case.copies[deep])
    grant = const(0)
    handoff = maximum(const(case.window), lane)
    done = shift(handoff, case.work[deep] * 100)
    yield done
    objects = {}
    for obj, duration, work in case.coupled:
        if obj not in objects:
            if (4 if deep else 1) + (len(objects) + 1) * 2 > case.capacity:
                grant = handoff
            lane = shift(maximum(lane, grant), 0, duration)
            objects[obj] = lane
        done = shift(maximum(done, objects[obj]), work * 100)
        yield done
    for ready, work in case.tail:
        done = shift(maximum(done, const(ready)), work * 100)
        yield done


def at(envelope, x):
    lines, knots = envelope
    a, b = lines[bisect_right(knots, x)]
    return a + b * x


class Gain:
    """Add completion differences via their affine slope changes."""

    def __init__(self):
        self.a = self.b = 0
        self.kinks = defaultdict(int)

    def add(self, shallow, deep):
        for sign, (lines, knots) in ((1, shallow), (-1, deep)):
            a, b = lines[0]
            self.a += sign * a
            self.b += sign * b
            for x, (_, y), (_, z) in zip(knots, lines, lines[1:]):
                self.kinks[x] += sign * (z - y)

    def bounds(self, lo, hi, remaining=0, last=None):
        knots = self.kinks.copy()
        a, b = self.a, self.b
        if remaining:
            for sign, (lines, breaks) in ((1, last[0]), (-1, last[1])):
                u, v = lines[0]
                a += remaining * sign * u
                b += remaining * sign * v
                for x, (_, y), (_, z) in zip(breaks, lines, lines[1:]):
                    knots[x] += remaining * sign * (z - y)
        for x, db in knots.items():
            if x <= lo:
                b += db
                a -= db * x
        extrema = [a + b * lo]
        for x in sorted(x for x, db in knots.items() if db and lo < x < hi):
            extrema.append(a + b * x)
            db = knots[x]
            b += db
            a -= db * x
        extrema.append(a + b * hi)
        return min(extrema), max(extrema)


def paired_stop(case, cost=0):
    """certified=1 for sign, 2 for regret at most the remaining information cost."""
    if not case.error and not cost:
        return selective(case, False)
    u, v = 100 - case.error, 100 + case.error
    gain = Gain()
    for seen, pair in enumerate(zip(affine_path(case, False), affine_path(case, True), strict=True), 1):
        gain.add(*pair)
        if seen <= len(case.coupled):
            continue
        lo, hi = gain.bounds(u, v)
        remain = case.length - seen
        if remain:
            lo2, hi2 = gain.bounds(u, v, remain, pair)
            lo, hi = min(lo, lo2), max(hi, hi2)
        if lo > 0 or hi <= 0:
            return lo > 0, seen, 2 * seen, 1
        if cost and min(-lo, hi) <= cost and remain:
            return lo + hi > 0, seen, 2 * seen, 2
    return lo + hi > 0, seen, 2 * seen, 0


class FifoBlock:
    """Independent suffix: each request finishes at prefix work plus max(input, threshold)."""

    def __init__(self, requests):
        work, threshold, self.thresholds = 0, None, []
        for ready, duration in requests:
            key = ready - work
            threshold = key if threshold is None else max(key, threshold)
            self.thresholds.append(threshold)
            work += duration
        self.sums = list(accumulate(self.thresholds, initial=0))
        self.work = work

    def gain(self, shallow, deep):
        def shifted(value):
            i = bisect_right(self.thresholds, value)
            return i * value + self.sums[-1] - self.sums[i]
        return shifted(shallow) - shifted(deep)


def block_point(case):
    """Charge the full block construction; keep FIFO order and fixed later actions."""
    block = FifoBlock((r * 100, p * 100) for r, p in case.tail)
    horizon = 1 + len(case.coupled)
    s, d = path(case, False, 100), path(case, True, 100)
    total = 0
    for _ in range(horizon):
        a, b = next(s), next(d)
        total += a - b
    return total + block.gain(a, b) > 0, horizon, 2 * horizon, 1


def partition_stop(case, bins=4):
    """Cheap paired envelope: opposite endpoint paths must share a cost subinterval."""
    if not case.error:
        return selective(case, False)
    assert (2 * case.error) % bins == 0
    width = 2 * case.error // bins
    starts = range(100 - case.error, 101 + case.error, width)
    ss = [path(case, False, x) for x in starts]
    dd = [path(case, True, x) for x in starts]
    lows, highs = [0] * bins, [0] * bins
    for seen, (s, d) in enumerate(zip(zip(*ss, strict=True), zip(*dd, strict=True), strict=True), 1):
        remain = case.length - seen
        lo, hi = float('inf'), -float('inf')
        for i in range(bins):
            l, h = s[i] - d[i + 1], s[i + 1] - d[i]
            lows[i] += l
            highs[i] += h
            lo = min(lo, lows[i] + min(0, remain * l))
            hi = max(hi, highs[i] + max(0, remain * h))
        if seen > len(case.coupled) and (lo > 0 or hi <= 0):
            return lo > 0, seen, 2 * len(ss) * seen, 1
    return lo + hi > 0, seen, 2 * len(ss) * seen, 0


def audit():
    # Only C is examined by the one-peer policy. D alone tips the comparison;
    # sharing D's object with C removes the admission edge.
    case = Case('unique', 0, 'dense', 6, 12, (1, 1), (5, 3), ((1, 2, 1), (2, 11, 1)), (), 100)
    def scores(c):
        return [tuple(t // 100 for t in path(c, d, 100)) for d in (False, True)]
    assert scores(case) == [(17, 18, 19), (15, 16, 24)]
    assert fixed(case, 2)[0] and not fixed(case, 3)[0]
    assert scores(replace(case, coupled=((1, 2, 1), (1, 2, 1)))) == [(17, 18, 19), (15, 16, 17)]
    relay = replace(case, coupled=((1, 2, 1), (2, 9, 1)), tail=((0, 1),))
    idle = replace(relay, tail=((22, 1),))
    assert scores(relay) == [(17, 18, 19, 20), (15, 16, 22, 23)]
    assert fixed(relay, 3)[0] and not fixed(relay, 4)[0] and fixed(idle, 4)[0]
    # Endpoints of the shared cost range can agree even when the interior disagrees.
    crossing = hull(((0, 0), (-100, 2)))
    g = Gain()
    g.add(crossing, (((-40, 1),), ()))
    assert g.bounds(0, 100) == (-10, 40)
    checks = 0
    for items in product(product(range(3), (1, 2)), repeat=3):
        block = FifoBlock(items)
        for a, b in product(range(6), repeat=2):
            def slow(t):
                total = 0
                for ready, work in items:
                    t = max(t, ready) + work
                    total += t
                return total
            assert block.gain(a, b) == slow(a) - slow(b)
            checks += 1
    for c in sample(7630, 12, holdout=False):
        arms = [list(affine_path(c, arm)) for arm in (False, True)]
        gain = Gain()
        for pair in zip(*arms, strict=True):
            gain.add(*pair)
        low, high = gain.bounds(100 - c.error, 100 + c.error)
        for x in range(100 - c.error, 101 + c.error):
            s, d = [list(path(c, arm, x)) for arm in (False, True)]
            assert [at(q, x) for q in arms[0]] == s and [at(q, x) for q in arms[1]] == d
            assert low <= sum(s) - sum(d) <= high
            checks += 1
        assert block_point(c)[0] == fixed(c, c.length)[0]
    print(f'{checks} independent FIFO-block and shared-cost exhaustive grid checks')


def study(seed, rounds, holdout):
    index = defaultdict(list)
    for c in sample(seed, rounds, holdout=holdout):
        s, d = list(path(c, False, c.actual_scale)), list(path(c, True, c.actual_scale))
        ssum, dsum = sum(s), sum(d)
        base_deep = fixed(c, 1)[0]
        strategies = {
            'target': lambda: fixed(c, 1),
            'fixed4': lambda: fixed(c, min(5, c.length)),
            'full_point': lambda: fixed(c, c.length),
            'block_point': lambda: block_point(c),
            'separate_ranges': lambda: selective(c, True),
            'partition_ranges': lambda: partition_stop(c),
            'shared_range': lambda: paired_stop(c),
            'shared_regret_0.25ms': lambda: paired_stop(c, cost=25),
        }
        for name, dispatch in strategies.items():
            start = perf_counter_ns()
            deep, seen, _, cert = dispatch()
            ns = perf_counter_ns() - start
            if name == 'target':
                baseline_ns = ns
            regret = (dsum if deep else ssum) - min(ssum, dsum)
            if name == 'block_point':
                assert deep == fixed(c, c.length)[0]
            if ('range' in name or 'regret' in name) and cert:
                assert regret <= (25 if cert == 2 else 0)
            base = d if base_deep else s
            candidate = d if deep else s
            index[c.share, c.load, c.error, name].append(
                (seen, c.length, cert, regret / 100, (sum(base) - sum(candidate)) / 100,
                 max(0, max(q-p for p, q in zip(base, candidate, strict=True))) / 100, ns,
                 (sum(base) - sum(candidate)) / 100 - c.length * (ns - baseline_ns) / 1e6))
    for (share, load, error, name), z in sorted(index.items()):
        n = len(z)
        yield dict(split='holdout' if holdout else 'calibration', sharing=share, load=load, error=error,
                   policy=name, n=n, mean_seen=sum(x[0] for x in z) / n, full=sum(x[0] == x[1] for x in z),
                   sign=sum(x[2] == 1 for x in z) if 'range' in name or 'regret' in name else '',
                   budget_stop=sum(x[2] == 2 for x in z), mean_regret_ms=sum(x[3] for x in z) / n,
                   mean_gain_ms=sum(x[4] for x in z) / n, worst_peer_ms=max(x[5] for x in z),
                   mean_decision_us=sum(x[6] for x in z) / n / 1000,
                   mean_gain_less_full_queue_delay_ms=sum(x[7] for x in z) / n)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rounds', type=int, default=150)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    audit()
    if a.out:
        rows = [*study(7630, a.rounds, False), *study(7631, a.rounds, True)]
        with a.out.open('x', newline='', encoding='utf-8') as out:
            w = csv.DictWriter(out, fieldnames=rows[0])
            w.writeheader()
            w.writerows(rows)


if __name__ == '__main__':
    main()
