"""Validate a marginal-externality certificate for stopping lookahead."""

import argparse
import csv
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from random import Random


@dataclass(frozen=True)
class Case:
    prefix_gap: int
    capacity: tuple[int, ...]
    queue: tuple[int, ...]
    cost: tuple[int, ...]
    actual: tuple[int, ...]

    @property
    def budget(self):
        return tuple(a + b + c for a, b, c in zip(self.capacity, self.queue, self.cost))


def certify(gap, budgets):
    bound = sum(budgets)
    if gap > bound:
        return 1, bound
    if gap + bound <= 0:
        return -1, bound
    return 0, bound


def exhaustive_audit():
    checked = 0
    for gap in range(-12, 13):
        for budgets in product(product(range(3), repeat=3), repeat=3):
            b = tuple(sum(x) for x in zip(*budgets))
            sign, bound = certify(gap, b)
            for actual in product(*[range(-x, x + 1) for x in b]):
                total = gap + sum(actual)
                if sign == 1:
                    assert total > 0, (gap, b, actual, total)
                if sign == -1:
                    assert total <= 0, (gap, b, actual, total)
                checked += 1
    print(f'{checked} bounded externality combinations; no certified sign was false')


def sample(seed, rounds, holdout):
    rng = Random(seed)
    for _ in range(rounds):
        n = rng.randrange(1, 4 if not holdout else 8)
        components = []
        actual = []
        for _ in range(n):
            b = tuple(rng.randrange(0, 5 if not holdout else 12) for _ in range(3))
            components.append(b)
            actual.append(rng.randint(-sum(b), sum(b)))
        yield Case(rng.randrange(-20, 21), tuple(x[0] for x in components),
                   tuple(x[1] for x in components), tuple(x[2] for x in components), tuple(actual))


def evaluate(seed, rounds, holdout):
    rows = []
    for i, case in enumerate(sample(seed, rounds, holdout)):
        budgets = case.budget
        sign, bound = certify(case.prefix_gap, budgets)
        actual = case.prefix_gap + sum(case.actual)
        rows.append(dict(split='holdout' if holdout else 'calibration', case=i,
                         requests=len(budgets), prefix_gap=case.prefix_gap,
                         capacity_bound=sum(case.capacity), queue_bound=sum(case.queue),
                         cost_bound=sum(case.cost), total_bound=bound,
                         actual_gain=actual, certified=sign,
                         certified_correct=int(sign == 0 or (sign > 0) == (actual > 0)),
                         missed_opportunity=int(sign == 0 and actual > 0)))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rounds', type=int, default=1000)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    exhaustive_audit()
    rows = [*evaluate(7650, a.rounds, False), *evaluate(7651, a.rounds, True)]
    with a.out.open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    for split in ('calibration', 'holdout'):
        z = [r for r in rows if r['split'] == split]
        print(split, len(z), 'certified', sum(r['certified'] != 0 for r in z),
              'correct', sum(r['certified_correct'] for r in z),
              'missed', sum(r['missed_opportunity'] for r in z))


if __name__ == '__main__':
    main()
