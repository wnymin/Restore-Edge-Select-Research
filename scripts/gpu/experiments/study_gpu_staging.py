"""Validate and summarize paired staging quota trials and independent holdout."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import median, mean


def read(path):
    cells = defaultdict(dict)
    settings = set()
    with path.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if int(r['rep']) < 0:
                continue
            settings.add((r['source'], int(r.get('retain_ms', '0'))))
            key = (r['topology'], int(r['quota_mib']), r['mode'], int(r['deep_work_ms']), int(r['rep']))
            if r['arm'] in cells[key]:
                raise ValueError(f'Duplicate {key}')
            capacity, target, peer = (int(r[k]) for k in ('quota_mib', 'target_mib', 'peer_mib'))
            eligible = target <= capacity and (key[0] != 'shared' or peer <= capacity)
            wait = key[0] == 'shared' and target + peer > capacity
            status = 'inadmissible' if not eligible else 'wait' if wait else 'granted'
            if r['outcome'] != status:
                raise ValueError(f'Wrong admission: {r}')
            if eligible:
                expected = max(target, peer) if wait else target+peer
                if int(r['peak_mib']) != expected or int(r['copied_bytes']) != (target+peer)*(1 << 20):
                    raise ValueError(f'Invalid peak or copied bytes: {r}')
                if key[0] == 'shared' and expected > capacity or key[2] == 'serial' and (
                        float(r['target_overlap_ms']) or float(r['peer_overlap_ms'])):
                    raise ValueError(f'Violated pool or serial invariant: {r}')
                if r.get('target_ready_ms'):
                    ready, prior, consume, end, cont = (float(r[k]) for k in ('target_ready_ms', 'prior_work_ms',
                        'target_consume_ms', 'target_device_ms', 'continuation_ms'))
                    if consume + .10 < max(ready, prior) or abs(end-consume-cont) > .10:
                        raise ValueError(f'Invalid observed critical path: {r}')
            cells[key][r['arm']] = r
    by_case = defaultdict(list)
    for (*key, _), arms in sorted(cells.items()):
        if set(arms) != {'shallow', 'deep'}:
            raise ValueError(f'Incomplete pair: {key}')
        by_case[tuple(key)].append((arms['shallow'], arms['deep']))
    if len(settings) != 1:
        raise ValueError(f'Expected one source and retention setting in {path}')
    print(path, *next(iter(settings)))
    return by_case, settings


def delta(pair):
    a, b = pair
    return float(b['target_device_ms'])-float(a['target_device_ms']) if b['outcome'] != 'inadmissible' else None


def show(data):
    print('topology quota mode     deepwork    n    D-S med  D wins   wait med  target D overlap  peer D overlap  peer D-S')
    for (top, cap, mode, work), pairs in data.items():
        ds = [delta(x) for x in pairs if delta(x) is not None]
        if not ds:
            print(f'{top:8} {cap:5} {mode:8} {work:8} {len(pairs):4}  deep inadmissible')
            continue
        vals = [median(float(b[k]) for _, b in pairs)
                for k in ('admission_wait_ms', 'target_overlap_ms', 'peer_overlap_ms')]
        peer = median(float(b['peer_done_ms'])-float(a['peer_done_ms']) for a, b in pairs)
        print(f'{top:8} {cap:5} {mode:8} {work:8} {len(ds):4} {median(ds):10.3f} {sum(d < 0 for d in ds):7}',
              *(f'{x:13.3f}' for x in [*vals, peer]))
        if pairs[0][1].get('target_ready_ms'):
            def ideal(r):
                t, p, c = (float(r[k]) for k in ('target_ready_ms', 'prior_work_ms', 'continuation_ms'))
                return max(t, p) + c
            predicted = [ideal(b)-ideal(a) for a,b in pairs]
            error = [abs(p - d) for p,d in zip(predicted, ds, strict=True)]
            print(f'  Critical path with observed device events: D-S median {median(predicted):+.3f}, '
                  f'correct sign {sum((p < 0) == (d < 0) for p,d in zip(predicted, ds))}/{len(ds)}, '
                  f'worst absolute residual {max(error):.3f} ms')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('calibration', type=Path)
    p.add_argument('--holdout', type=Path)
    args = p.parse_args()
    cal, settings = read(args.calibration)
    show(cal)
    if not args.holdout:
        return
    holdout, other = read(args.holdout)
    if args.holdout.resolve() == args.calibration.resolve() or holdout.keys() != cal.keys() or other != settings:
        p.error('Use separate calibration and holdout with the same cells')
    show(holdout)
    selected = {key for key, items in cal.items() if all(delta(x) is not None and delta(x) < 0 for x in items)}
    print('Calibration cells selecting deep:', sorted(selected))
    for policy in ('fixed shallow', 'deep if admissible', 'calibrated'):
        vals = [delta(pair) if delta(pair) is not None and (policy == 'deep if admissible'
                or policy == 'calibrated' and key in selected) else 0 for key, items in holdout.items() for pair in items]
        print(policy, f'mean device difference: {mean(vals):+.3f} ms; worse than shallow: {sum(v > 0 for v in vals)}/{len(vals)}')


if __name__ == '__main__':
    main()
