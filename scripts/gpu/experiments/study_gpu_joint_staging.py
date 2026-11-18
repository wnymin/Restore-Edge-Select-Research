"""Validate paired CUDA staging evidence and evaluate the frozen window rule."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import median


def read(path):
    cells = defaultdict(lambda: defaultdict(dict))
    with path.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            rep = int(row['rep'])
            if rep < 0:
                continue
            mode, cap, shallow = row['mode'], int(row['quota_mib']), int(row['shallow_work_ms'])
            arm = row['arm']
            assert arm in ('shallow', 'deep') and arm not in cells[mode, cap, shallow][rep]
            size = 48 if arm == 'deep' else 16
            union = size + (64 if mode == 'distinct' else 32)
            blocked = union > cap
            assert size + 32 <= cap and int(row['target_mib']) == size
            assert int(row['union_mib']) == int(row['copied_mib']) == union
            assert int(row['peak_mib']) == (max(size + 32, 64) if blocked else union)
            assert int(row['blocked']) == blocked
            if blocked:
                assert float(row['peer_d_ready_ms']) > float(row['release_ms'])
            cells[mode, cap, shallow][rep][arm] = row
    assert len(cells) == 6 and all(len(arms) == 2 for reps in cells.values() for arms in reps.values())
    windows = {int(row['nominal_window_ms']) for reps in cells.values()
               for arms in reps.values() for row in arms.values()}
    assert len(windows) == 1
    return cells, windows.pop()


def score(row):
    return sum(float(row[k] or 0) for k in ('target_done_ms', 'peer_c_done_ms', 'peer_d_done_ms'))


def estimate(mode, cap, work, window, d_transfer, arm, blocked_transfer=None):
    size = 48 if arm == 'deep' else 16
    target_done = window + work - (2 if arm == 'deep' else 0)
    c_done = target_done + 2 if mode != 'd-only' else 0
    if mode == 'c-only':
        return target_done + c_done
    blocked = mode == 'distinct' and size + 64 > cap
    if mode == 'alias':
        d_ready = 0
    elif blocked:
        d_ready = window + (blocked_transfer if blocked_transfer is not None else d_transfer)
    else:
        d_ready = d_transfer
    return target_done + c_done + max(target_done, c_done, d_ready) + 1


def show(label, cells, window, transfer, blocked_transfer=None):
    print(label, f'nominal window {window} ms; calibration gives {transfer:.3f} ms for D read+upload')
    print('mode cap shallow_work n D-S_sum_median deep_wins D-S_target_median D_D-S_median D_deep_wait_median')
    errors = defaultdict(list)
    for (mode, cap, work), reps in cells.items():
        rules = {'pairwise': 'deep', 'union': 'shallow' if mode == 'distinct' and cap < 112 else 'deep',
                 'window': min(('shallow', 'deep'), key=lambda arm: estimate(mode, cap, work, window, transfer, arm))}
        if blocked_transfer is not None:
            rules['window with separate grant estimate'] = min(('shallow', 'deep'), key=lambda arm: estimate(
                mode, cap, work, window, transfer, arm, blocked_transfer))
        rows = list(reps.values())
        diffs = [score(r['deep']) - score(r['shallow']) for r in rows]
        target = [float(r['deep']['target_done_ms']) - float(r['shallow']['target_done_ms']) for r in rows]
        d = [float(r['deep']['peer_d_done_ms'] or 0) - float(r['shallow']['peer_d_done_ms'] or 0) for r in rows]
        print(mode, cap, work, len(rows), f'{median(diffs):+.3f}', sum(v < 0 for v in diffs),
              f'{median(target):+.3f}', f'{median(d):+.3f}', f"{median(float(r['deep']['wait_ms']) for r in rows):.3f}")
        for rule, arm in rules.items():
            errors[rule].extend(max(0, v if arm == 'deep' else -v) for v in diffs)
    for rule, losses in errors.items():
        print(rule, f'nonzero paired loss {sum(v > 0 for v in losses)}/{len(losses)}; total {sum(losses):.3f} ms')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('calibration', type=Path)
    p.add_argument('holdout', type=Path)
    p.add_argument('--follow-up', type=Path)
    args = p.parse_args()
    cal, cal_window = read(args.calibration)
    test, test_window = read(args.holdout)
    assert cal.keys() == test.keys() and test_window != cal_window
    transfer = median(float(row['ssd_read_ms']) + float(row['ssd_upload_ms'])
                      for arms in cal['d-only', 80, 7].values() for row in arms.values())
    show('calibration', cal, cal_window, transfer)
    show('holdout', test, test_window, transfer)
    if args.follow_up:
        follow, follow_window = read(args.follow_up)
        assert cal.keys() == follow.keys() and follow_window not in (cal_window, test_window)
        blocked_transfer = median(float(row['peer_d_ready_ms']) - float(row['release_ms'])
                                  for reps in cal.values() for arms in reps.values() for row in arms.values()
                                  if int(row['blocked']))
        print(f'Separate release-to-ready calibration: {blocked_transfer:.3f} ms')
        show('follow-up', follow, follow_window, transfer, blocked_transfer)


if __name__ == '__main__':
    main()
