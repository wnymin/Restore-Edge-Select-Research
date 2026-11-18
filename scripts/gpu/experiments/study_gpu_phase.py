"""Verify and summarize randomized paired phase interventions, then replay a frozen guard."""

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import median, mean


def pairs(files):
    groups = defaultdict(dict)
    for path in files:
        with path.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if int(row['rep']) < 0:
                    continue
                key = tuple(row[k] for k in ('shallow', 'deep', 'seed', 'rep', 'phase'))
                if row['arm'] in groups[key]:
                    raise ValueError(f'Duplicate trial {key}, {row["arm"]}')
                groups[key][row['arm']] = row
    for (lo, hi, seed, rep, phase), arms in sorted(groups.items()):
        if not {'shallow', 'deep'} <= arms.keys():
            raise ValueError(f'Incomplete block {(lo, hi, seed, rep, phase)}')
        shallow, deep = (arms[a] for a in ('shallow', 'deep'))
        if len({a['arm_order'] for a in arms.values()}) != len(arms):
            raise ValueError('Duplicate arm order')
        for row in arms.values():
            if phase == 'idle' and int(row['decode_n']) != 0 or phase != 'idle' and int(row['decode_n']) < 9:
                raise ValueError(f'Invalid peer metrics: {row}')
            if bool(row['restore_ms']) != (row.get('execution', row['arm']) == 'deep'):
                raise ValueError(f'Unexpected restore: {row}')
        def delta(field, before=shallow, after=deep):
            return float(after[field] or 0) - float(before[field] or 0)
        yield dict(case=(lo, hi, phase), seed=seed, rep=rep, target=delta('target_ms'),
                   peer=delta('peer_done_ms'), gap=delta('peer_max_gap_ms'), makespan=delta('makespan_ms'),
                   executor=deep.get('peer_executor', 'same'),
                   extra=int(deep.get('extra_decode_n') or 0), extra_done=delta('extra_done_ms') if 'extra_done_ms' in shallow else 0,
                   extra_gap=delta('extra_max_gap_ms') if 'extra_max_gap_ms' in shallow else 0,
                   token_match=len({a['target_token'] for a in arms.values()}) == 1,
                   release_diff=delta('release_from_peer_ms'),
                   restore=float(deep['restore_ms']),
                   same_depth=delta('target_ms', arms['resident']) if 'resident' in arms else None,
                   resident_gain=delta('target_ms', shallow, arms['resident']) if 'resident' in arms else None,
                   restore_vs_delayed=delta('target_ms', arms['delayed']) if 'delayed' in arms else None,
                   actual_guard={k: delta(f, shallow, arms['guard']) for k, f in
                                 [('target', 'target_ms'), ('peer', 'peer_done_ms'), ('gap', 'peer_max_gap_ms')]}
                   | {'deep': arms['guard']['execution'] == 'deep'} if 'guard' in arms else None)


def breakdown(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['case']].append(row)
    return groups


def show(groups):
    print('boundaries phase         N   target     peer      gap makespan  D wins  token  restore  vs res  res vs S  release')
    for (lo, hi, phase), rows in groups.items():
        values = []
        for f in ('target', 'peer', 'gap', 'makespan', 'restore', 'same_depth', 'resident_gain', 'release_diff'):
            x = [r[f] for r in rows if r[f] is not None]
            values.append(f'{median(x):8.2f}' if x else '       -')
        print(f'{lo:>4}->{hi:<5} {phase:8} {len(rows):4}', *values[:4],
              f'{sum(r["target"]<0 for r in rows):7}', f'{sum(r["token_match"] for r in rows):6}', *values[4:])
        if any(r['extra'] for r in rows):
            print('  extra decoder, D-S completion / longest token gap: '
                  f'{median(r["extra_done"] for r in rows):+.2f} / {median(r["extra_gap"] for r in rows):+.2f} ms')


def replay(rows, rules, cap, gap_cap, static):
    for policy in ('shallow', 'deep', 'pending', 'boundary', 'guard', 'live_guard'):
        if policy == 'live_guard':
            live = [r['actual_guard'] for r in rows if r['actual_guard'] is not None]
            if live:
                print(json.dumps(dict(policy=policy, deep=sum(r['deep'] for r in live), blocks=len(live),
                      target_mean=mean(r['target'] for r in live), max_peer=max(r['peer'] for r in live),
                      max_gap=max(r['gap'] for r in live), slow_target=sum(r['target'] > 0 for r in live),
                      violations=sum(r['peer'] > cap or r['gap'] > gap_cap for r in live))))
            continue
        selected = [r for r in rows if policy == 'deep' or (policy == 'guard' and r['case'] in rules)
                    or (policy == 'boundary' and r['case'][:2] in static)
                    or (policy == 'pending' and r['case'][2] != 'idle')]
        target = [r['target'] if r in selected else 0 for r in rows]
        peer = [r['peer'] if r in selected else 0 for r in rows]
        gap = [r['gap'] if r in selected else 0 for r in rows]
        regret = [v - (min(0, r['target']) if r['peer'] <= cap and r['gap'] <= gap_cap else 0)
                  for r, v in zip(rows, target)]
        print(json.dumps(dict(policy=policy, deep=len(selected), blocks=len(rows), target_mean=mean(target),
              max_peer=max(peer), max_gap=max(gap), unconstrained_target_regret=mean(
                  v - min(0, r['target']) for r, v in zip(rows, target)),
              compliant_regret=mean(v for v, p, g in zip(regret, peer, gap) if p <= cap and g <= gap_cap)
                  if any(p <= cap and g <= gap_cap for p, g in zip(peer, gap)) else None,
              violations=sum(p > cap or g > gap_cap for p, g in zip(peer, gap)),
              slow_target=sum(v > 0 for v in target))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('calibration', type=Path, nargs='+')
    p.add_argument('--holdout', type=Path, nargs='+')
    p.add_argument('--executor-control', type=Path, help='Same prompts run with peer on the other executor topology')
    p.add_argument('--load-control', type=Path, help='Same prompts with a third active decoder')
    p.add_argument('--min-target', type=float, default=20)
    p.add_argument('--peer-cap', type=float, default=50)
    p.add_argument('--gap-cap', type=float, default=50)
    p.add_argument('--freeze', type=Path)
    p.add_argument('--frozen', type=Path, help='Verify a recorded rule against calibration and actual guard arms')
    args = p.parse_args()
    rows = list(pairs(args.calibration))
    groups = breakdown(rows)
    if not rows:
        p.error('No measured pairs in calibration')
    show(groups)
    if args.executor_control or args.load_control:
        if args.executor_control and args.load_control:
            p.error('Compare only one control at a time')
        other = list(pairs([args.executor_control or args.load_control]))
        show(breakdown(other))
        keyed = {(r['case'], r['seed'], r['rep']): r for r in other}
        if len(keyed) != len(rows) or keyed.keys() != {(r['case'], r['seed'], r['rep']) for r in rows}:
            p.error('Executor comparison must use the same phase blocks and seeds')
        diff = defaultdict(list)
        for a in rows:
            b = keyed[a['case'], a['seed'], a['rep']]
            if args.executor_control and {a['executor'], b['executor']} != {'same', 'isolated'}:
                p.error('Expected same and isolated executor captures')
            if args.load_control and (a['extra'] or not b['extra']):
                p.error('Expected the extra decoder only in the control')
            diff[a['case']].append((a['target']-b['target']) if args.executor_control else (b['target']-a['target']))
        for key, d in diff.items():
            label = 'primary minus control' if args.executor_control else 'third decoder minus base'
            print(key, f'{label} effect {median(d):+.2f} ms; positive {sum(x>0 for x in d)}/{len(d)}')
    rules = {key for key, group in groups.items() if len(group) >= 4 and all(
        r['target'] < -args.min_target and r['peer'] <= args.peer_cap and r['gap'] <= args.gap_cap for r in group)}
    record = dict(deep_cases=[list(k) for k in sorted(rules)], min_target_ms=args.min_target,
                  peer_cap_ms=args.peer_cap, gap_cap_ms=args.gap_cap,
                  calibration_sha256={x.name: sha256(x.read_bytes()).hexdigest() for x in args.calibration})
    if args.freeze:
        with args.freeze.open('x', encoding='utf-8') as out:
            json.dump(record, out, indent=2)
            out.write('\n')
    if args.frozen and json.loads(args.frozen.read_text(encoding='utf-8')) != record:
        p.error('Frozen rule does not match the calibration files or settings')
    if args.holdout:
        holdout = list(pairs(args.holdout))
        if not holdout or {r['seed'] for r in holdout} & {r['seed'] for r in rows}:
            p.error('Holdout must have measurements and distinct prompt seeds')
        if any(r['actual_guard'] is not None and r['actual_guard']['deep'] != (r['case'] in rules) for r in holdout):
            p.error('A live arm used an action different from the frozen rule')
        print('Frozen deep cases:', sorted(rules))
        show(breakdown(holdout))
        static = {k[:2] for k in groups if all(other in rules for other in groups if other[:2] == k[:2])}
        replay(holdout, rules, args.peer_cap, args.gap_cap, static)


if __name__ == '__main__':
    main()
