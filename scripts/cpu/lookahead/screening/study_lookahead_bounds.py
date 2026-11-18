"""Measure the value of another visible FIFO request under bounded cost error."""

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter_ns


@dataclass(frozen=True)
class Case:
    share: str
    error: int
    load: str
    capacity: int
    window: int
    copies: tuple[int, int]
    work: tuple[int, int]
    # Each direct peer has one object ID, a nominal transfer, and a continuation.
    coupled: tuple[tuple[int, int, int], ...]
    tail: tuple[tuple[int, int], ...]
    actual_scale: int

    @property
    def length(self):
        return 1 + len(self.coupled) + len(self.tail)


def path(case, deep, scale):
    """Known FIFO order; correlated transfer multiplier, fixed tail readiness."""
    assert len({obj for obj, _, _ in case.coupled}) * 2 <= case.capacity
    target_size = 4 if deep else 1
    lane = case.copies[deep] * 100
    handoff = max(case.window * 100, lane)
    done = handoff + case.work[deep] * 100
    yield done
    objects = {}
    grant = 0
    for obj, duration, work in case.coupled:
        if obj not in objects:
            if target_size + (len(objects) + 1) * 2 > case.capacity:
                grant = handoff
            lane = max(lane, grant) + duration * scale
            objects[obj] = lane
        done = max(done, objects[obj]) + work * 100
        yield done
    for ready, work in case.tail:
        done = max(done, ready * 100) + work * 100
        yield done


def stop_interval(prefix_low, prefix_high, gap_low, gap_high, remaining):
    """Conditional on equal tail readiness/work in both branches."""
    return (prefix_low + min(0, remaining * gap_low),
            prefix_high + max(0, remaining * gap_high))


def fixed(case, horizon, scale=100):
    gains = zip(path(case, False, scale), path(case, True, scale), strict=True)
    gain = 0
    for seen in range(horizon):
        s, d = next(gains)
        gain += s - d
    return gain > 0, horizon, 2 * horizon, True


def selective(case, use_error):
    # A shared cost multiplier is monotone in every completion, for fixed reservation order.
    length = case.length
    minimum = 1 + len(case.coupled)
    if not use_error or case.error == 0:
        gain = 0
        for seen, (s, d) in enumerate(zip(path(case, False, 100), path(case, True, 100), strict=True), 1):
            gap = s - d
            gain += gap
            if seen >= minimum:
                lower, upper = stop_interval(gain, gain, gap, gap, length - seen)
                if lower > 0 or upper <= 0:
                    return lower > 0, seen, 2 * seen, True
    else:
        low, high = 100 - case.error, 100 + case.error
        states = (path(case, False, low), path(case, True, high), path(case, False, high), path(case, True, low))
        lo, hi = 0, 0
        for seen, (slo, dhi, shi, dlo) in enumerate(zip(*states, strict=True), 1):
            low_gap, high_gap = slo - dhi, shi - dlo
            lo += low_gap
            hi += high_gap
            if seen >= minimum:
                lower, upper = stop_interval(lo, hi, low_gap, high_gap, length - seen)
                if lower > 0 or upper <= 0:
                    return lower > 0, seen, 4 * seen, True
    # The true sign is unresolved even after all visible requests. The smaller
    # interval bound on regret chooses deep iff -lower < upper (ties go shallow).
    return lower + upper > 0, seen, 4 * seen, False


def sample(seed, rounds, *, holdout):
    rng = Random(seed)
    for share, error, load in product(('unique', 'alias'), (0, 10, 30, 50), ('dense', 'slack')):
        for _ in range(rounds):
            window = rng.randrange(8, 28) if not holdout else rng.randrange(15, 37)
            copies = (rng.randrange(1, 5), rng.randrange(4, 17))
            work = rng.randrange(3, 12)
            coupled = []
            unique = 0
            for i in range(rng.randrange(1, 4)):
                new = i == 0 or share == 'unique' or rng.randrange(3) == 0
                if new:
                    unique += 1
                coupled.append((unique, rng.randrange(3, 18), rng.randrange(1, 5)))
            capacity = rng.randrange(max(4, unique * 2), min(12, 5 + unique * 2))
            n_tail = rng.randrange(2, 12) if not holdout else rng.randrange(8, 29)
            tick = window + rng.randrange(0, 10)
            tail = []
            for _ in range(n_tail):
                tick += rng.randrange(0, 4) if load == 'dense' else rng.randrange(3, 12)
                tail.append((tick, rng.randrange(1, 5)))
            yield Case(share, error, load, capacity, window, copies, (work, work - rng.randrange(1, 4)),
                       tuple(coupled), tuple(tail), rng.randint(100 - error, 100 + error))


POLICIES = {
    'target': lambda c: fixed(c, 1),
    'fixed-4': lambda c: fixed(c, min(5, c.length)),
    'fixed-all': lambda c: fixed(c, c.length),
    'point-stop': lambda c: selective(c, False),
    'interval-stop': lambda c: selective(c, True),
}


def evaluate(cases, group):
    for i, case in enumerate(cases):
        actual = [list(path(case, arm, case.actual_scale)) for arm in (False, True)]
        score = [sum(branch) for branch in actual]
        full = fixed(case, case.length)[0]
        local = fixed(case, 1)[0]
        for label, policy in POLICIES.items():
            start = perf_counter_ns()
            deep, seen, steps, resolved = policy(case)
            elapsed = perf_counter_ns() - start
            if label == 'point-stop':
                assert deep == full
            if label == 'interval-stop' and resolved:
                assert score[deep] == min(score)
            yield dict(group=group, case=i, share=case.share, cost_error_pct=case.error,
                       load=case.load, requests=case.length, policy=label, deep=int(deep),
                       seen=seen, updates=steps, decision_ns=elapsed, resolved=int(resolved),
                       oracle_sum_ms=min(score) / 100, regret_ms=(score[deep] - min(score)) / 100,
                       benefit_over_target_ms=(score[local] - score[deep]) / 100,
                       max_degradation_from_target_ms=max(0, max(x - y for x, y in zip(
                           actual[deep], actual[local], strict=True))) / 100)


def summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['group'], row['policy']].append(row)
    for (group, policy), cell in groups.items():
        n = len(cell)
        avg = lambda k: sum(r[k] for r in cell) / n
        print(group, policy, n, f"seen={avg('seen'):.2f}, updates={avg('updates'):.2f}, "
              f"time={median(r['decision_ns'] for r in cell)/1000:.2f} us, "
              f"regret={avg('regret_ms'):.2f} ms, gain={avg('benefit_over_target_ms'):.2f} ms, "
              f"worst={max(r['max_degradation_from_target_ms'] for r in cell):.2f} ms, "
              f"resolved={avg('resolved'):.1%}")


def write_cells(rows, out):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ('group', 'share', 'load', 'cost_error_pct', 'policy'))
        groups[key].append(row)
    with out.open('x', newline='', encoding='utf-8') as stream:
        fields = ('group', 'share', 'load', 'cost_error_pct', 'policy', 'cases', 'mean_seen', 'mean_updates',
                  'mean_regret_ms', 'mean_gain_ms', 'worst_degradation_ms', 'resolved_count', 'full_horizon_count')
        writer = csv.writer(stream)
        writer.writerow(fields)
        for key, cell in sorted(groups.items()):
            n = len(cell)
            avg = lambda k: round(sum(r[k] for r in cell) / n, 6)
            writer.writerow([*key, n, avg('seen'), avg('updates'), avg('regret_ms'), avg('benefit_over_target_ms'),
                             max(r['max_degradation_from_target_ms'] for r in cell), sum(r['resolved'] for r in cell),
                             sum(r['seen'] == r['requests'] for r in cell)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rounds', type=int, default=100)
    p.add_argument('--out', type=Path)
    p.add_argument('--cells', type=Path)
    args = p.parse_args()
    if args.rounds < 1:
        p.error('Positive round count required')
    rows = [*evaluate(sample(7610, args.rounds, holdout=False), 'calibration'),
            *evaluate(sample(7611, args.rounds, holdout=True), 'holdout')]
    summary(rows)
    if args.cells:
        write_cells(rows, args.cells)
    if args.out:
        with args.out.open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
