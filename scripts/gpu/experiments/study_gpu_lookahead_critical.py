"""Recompute critical-window comparisons from archived paired device events."""

import argparse
from collections import defaultdict
import csv
from statistics import mean, median


def rows(path):
    with open(path, newline='', encoding='utf-8') as source:
        return [r for r in csv.DictReader(source) if int(r['rep']) >= 0]


def delay(row):
    return float(row['peer_d_ready_ms']) - float(row['release_ms'])


def total(row):
    return sum(float(row[f'{who}_done_ms']) for who in ('target', 'peer_c', 'peer_d'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('holdout')
    p.add_argument('calibration', nargs='+')
    a = p.parse_args()
    calibration = [delay(r) for path in a.calibration for r in rows(path) if r['blocked'] == '1']
    lo, point, hi = min(calibration), median(calibration), max(calibration)
    print(f'Calibration: n={len(calibration)}, low/median/high={lo:.3f}/{point:.3f}/{hi:.3f} ms')
    pairs = defaultdict(dict)
    for r in rows(a.holdout):
        assert r['mode'] == 'distinct' and r['quota_mib'] == '80'
        pair = pairs[int(r['shallow_work_ms']), int(r['rep'])]
        assert r['arm'] not in pair
        pair[r['arm']] = r
    cases = defaultdict(list)
    for (work, _), pair in pairs.items():
        s, d = pair['shallow'], pair['deep']
        assert s['blocked'] == '0' and d['blocked'] == '1'
        assert float(s['peer_d_ready_ms']) < float(s['peer_c_done_ms'])
        gain = total(s) - total(d)
        observed = delay(d)
        model = 6 - max(0, observed - work)
        prefer_deep = point < work + 6
        cases[work].append((gain, model, observed, prefer_deep))
    print('work,n,deep_wins,median_gain_ms,point_correct,point_regret_ms')
    for work, z in sorted(cases.items()):
        print(work, len(z), sum(g > 0 for g, *_ in z), f'{median(g for g, *_ in z):.3f}',
              sum((g > 0) == p for g, _, _, p in z), f'{mean(max(0, -g if p else g) for g, _, _, p in z):.3f}', sep=',')
    z = [item for cell in cases.values() for item in cell]
    certified = [(g, work + 6 > hi) for work, cell in cases.items() for g, *_ in cell
                 if work + 6 < lo or work + 6 > hi]
    print(f'All deep regret: {mean(max(0, -g) for g, *_ in z):.3f} ms per pair')
    print(f'Point regret: {mean(max(0, -g if p else g) for g, _, _, p in z):.3f} ms per pair')
    print(f'Empirical range misses: {sum(not lo <= v <= hi for _, _, v, _ in z)}/{len(z)}')
    print(f'Range excludes zero: {len(certified)}/{len(z)}, correct: {sum((g > 0) == p for g,p in certified)}')
    print(f'Post-hoc sign matches: {sum((g > 0) == (m > 0) for g,m,_,_ in z)}/{len(z)}; '
          f'median absolute residual {median(abs(g-m) for g,m,_,_ in z):.3f} ms')


if __name__ == '__main__':
    main()
