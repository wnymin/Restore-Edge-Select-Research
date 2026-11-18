"""Factorial offline ceiling for knowing one more peer or the exact common cost."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path

from scripts.cpu.lookahead.screening.study_lookahead_bounds import fixed, path, sample


def run(seed, rounds, holdout):
    cells = defaultdict(list)
    for c in sample(seed, rounds, holdout=holdout):
        arms = [list(path(c, arm, c.actual_scale)) for arm in (False, True)]
        sums = [sum(a) for a in arms]
        target = fixed(c, 1)[0]
        strategies = {
            'first_peer_nominal': fixed(c, 2)[0],
            'all_visible_nominal': fixed(c, c.length)[0],
            'first_peer_exact_cost': fixed(c, 2, c.actual_scale)[0],
            'all_visible_exact_cost': fixed(c, c.length, c.actual_scale)[0],
        }
        for label, deep in strategies.items():
            loss = sums[deep] - min(sums)
            assert loss >= 0 and (label != 'all_visible_exact_cost' or not loss)
            cells[c.share,c.load,c.error,label].append(
                (loss, max(0, max(a-b for a,b in zip(arms[deep],arms[target],strict=True)))))
    for (share,load,error,label), samples in sorted(cells.items()):
        n=len(samples)
        yield dict(split='holdout' if holdout else 'calibration', sharing=share, load=load, error=error, action=label,
                   cases=n, choices_with_regret=sum(l>0 for l,_ in samples),
                   mean_regret_ms=sum(l for l,_ in samples)/n/100,
                   worst_single_request_vs_target_ms=max(w for _,w in samples)/100)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a=p.parse_args()
    cells=[*run(7640,150,False),*run(7641,150,True)]
    with a.out.open('x',newline='',encoding='utf-8') as file:
        w=csv.DictWriter(file,fieldnames=cells[0])
        w.writeheader()
        w.writerows(cells)


if __name__ == '__main__':
    main()
