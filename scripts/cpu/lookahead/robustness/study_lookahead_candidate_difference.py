"""Compare two fixed candidate plans by a common event-branch partition."""

import argparse
import csv
import json
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
from random import Random

from scripts.cpu.lookahead.robustness.study_lookahead_residual import Affine, Domain, partition, snapshots
from scripts.cpu.lookahead.screening.study_lookahead_workload_bounds import random_cases, clone, WorkloadBounds
from scripts.cpu.lookahead.screening.study_lookahead_trigger import cases
from scripts.cpu.lookahead.screening.study_lookahead_phase1 import Experiment, ROOT, read


class Toy:
    initial = set()
    window = 2
    budget = 2
    candidates = ({0: (), 1: ()}, {0: (), 1: ('D',)}, {0: (), 1: ('A', 'B')})
    lengths = (1, 2, 1)
    objects = {'A': dict(bytes=1, tier='RAM'), 'B': dict(bytes=1, tier='RAM'),
               'D': dict(bytes=3, tier='RAM')}
    _suffix_times = ({0: F(10), 1: F(10)}, {0: F(0), 1: F(5)}, {0: F(0), 1: F(0)})
    _transfer_times = {'A': F(1), 'B': F(4), 'D': F(1)}

    def size(self, objects):
        return sum(self.objects[o]['bytes'] for o in set(objects))

    def path_time(self, path, size):
        return size


def score_partition(exp, state, plan, lo, hi, estimate, speeds):
    return partition(exp, state, plan, estimate, speeds, lo, hi)


def affine_at(cells, x):
    for domain, ends, _ in cells:
        if domain.contains(x):
            return sum(ends, Affine()), domain
    raise AssertionError(x)


def common_partition(exp, state, left_plan, right_plan, lo, hi, estimate=None, speeds=None):
    estimate = estimate or {}
    speeds = speeds or dict(ram_to_device=F(1), ssd_to_device=F(1))
    left = score_partition(exp, state, left_plan, lo, hi, estimate, speeds)
    right = score_partition(exp, state, right_plan, lo, hi, estimate, speeds)
    points = {lo, hi}
    for cells in (left, right):
        for domain, _, _ in cells:
            points.add(domain.lo)
            points.add(domain.hi)
    points = sorted(points)
    rows = []
    for a, b in zip(points, points[1:]):
        if a == b:
            continue
        x = (a + b) / 2
        left_score, _ = affine_at(left, x)
        right_score, _ = affine_at(right, x)
        delta = left_score - right_score
        rows.append((Domain(a, b, False, False), delta))
    for x in points:
        left_score, _ = affine_at(left, x)
        right_score, _ = affine_at(right, x)
        rows.append((Domain(x, x, True, True), left_score - right_score))
    rows.sort(key=lambda row: (row[0].lo, row[0].hi, not row[0].left))
    for a, b in zip(rows, rows[1:]):
        assert a[0].hi == b[0].lo and a[0].right != b[0].left
    return rows, left, right


def extrema(cells):
    values = [score.at(x) for domain, score in cells for x in (domain.lo, domain.hi)]
    return min(values), max(values)


def audit(exp, state, plans, partitions, common, estimate, speeds):
    checked = 0
    # Original simulator is the independent oracle; check individual completion
    # vectors, rather than compare two algebraic transforms of symbolic output.
    for domain, delta in common:
        points = {domain.lo} if domain.lo == domain.hi else {
            (3*domain.lo+domain.hi)/4, (domain.lo+domain.hi)/2,
            (domain.lo+3*domain.hi)/4}
        if delta.b:
            root = -delta.a/delta.b
            if domain.contains(root):
                points.add(root)
        for x in points:
            changed = (*state[:6], (state[0]+x, state[6][1]), *state[7:])
            scores = []
            for plan, cells in zip(plans, partitions):
                result = Experiment._run(exp, 'fixed', 0, plan, 100, state[2],
                                         snapshot=clone(changed), estimate=estimate,
                                         forecast_speed=speeds, lean_rollout=True)
                cell = next(c for c in cells if c[0].contains(x))
                assert result['completion_ms'] == [t.at(x) for t in cell[1]]
                scores.append(sum(result['completion_ms']))
            assert scores[0]-scores[1] == delta.at(x)
            checked += 1
    return checked


def write_rows(rows, path):
    fields = ('seed', 'case', 'snapshot', 'pair', 'head', 'left_candidate', 'right_candidate',
              'lo', 'hi', 'left', 'right', 'intercept', 'slope', 'sign',
              'pair_lower', 'pair_upper', 'independent_lower', 'independent_upper',
              'workload_lower', 'workload_upper')
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def classify(delta, domain):
    left = delta.at(domain.lo)
    right = delta.at(domain.hi)
    if left == right == 0:
        return 'tie'
    if (left >= 0 and right >= 0 and
        not (left == 0 and domain.left) and not (right == 0 and domain.right)):
        return 'positive'
    if (left <= 0 and right <= 0 and
        not (left == 0 and domain.left) and not (right == 0 and domain.right)):
        return 'negative'
    return 'crosses-or-zero'


def toy_rows():
    exp = Toy()
    state = (F(0), 0, 3, {2: {'A', 'B'}}, set(), None, (F(1), 'A'), (), set(), None, {})
    rows = common_partition(exp, state, (0, 1, 1), (0, 0, 1), F(1, 1000), F(20))[0]
    assert len(rows) == 7
    for domain, delta in rows:
        points = {domain.lo} if domain.lo == domain.hi else {
            (3*domain.lo+domain.hi)/4, (domain.lo+domain.hi)/2,
            (domain.lo+3*domain.hi)/4}
        for x in points:
            # Hand-derived difference from the abstract three-request schedule.
            expected = F(12) if x <= 6 else 6+x if x < 10 else x-2
            assert delta.at(x) == expected
            assert sum(d.contains(x) for d,_ in rows) == 1
    assert classify(Affine(0,1), Domain(F(0),F(1),False,False)) == 'positive'
    assert classify(Affine(0,1), Domain(F(0),F(1),True,False)) == 'crosses-or-zero'
    assert classify(Affine(), Domain(F(0),F(1))) == 'tie'
    return rows


def real_rows(args):
    model = read(ROOT / 'configs/qwen3.5-4b.json')['models'][0]
    rng = Random(args.seed + 1)
    rows = []
    stats = Counter()
    source = (random_cases(model, args.seed, args.rounds) if args.split == 'random' else
              ((t, l, {'ssd_to_device':F(l['estimate'])}) for t,l in cases(args,model)))
    for case_index, (trace, labels, estimate) in enumerate(source):
        if stats['pairs'] >= args.max_pairs:
            break
        exp = WorkloadBounds(trace, model, labels['budget'])
        plan = tuple(rng.choice(choices) for choices in exp.eligible)
        speeds = {p: values[1][0] for p, values in exp.io_epochs.items()}
        for snapshot_index, state in enumerate(snapshots(exp, plan, estimate, speeds)):
            if stats['pairs'] >= args.max_pairs:
                break
            head = state[1]
            # Existing reservations commit a selected boundary. Do not silently
            # reinterpret such a checkpoint as a legal online re-selection.
            if head in state[3]:
                stats['skipped_committed_heads'] += 1
                continue
            choices = exp.eligible[head]
            if len(choices) < 2:
                continue
            base = plan[head]
            for alternate in choices:
                if stats['pairs'] >= args.max_pairs:
                    break
                if alternate == base:
                    continue
                left_plan = plan
                right_plan = tuple(alternate if j == head else c for j, c in enumerate(plan))
                residual = state[6][0] - state[0]
                lo, hi = F(1, 1000), max(F(64), residual * 4)
                common, left, right = common_partition(exp, state, left_plan, right_plan,
                                                       lo, hi, estimate, speeds)
                stats['audited_points'] += audit(exp, state, (left_plan,right_plan),
                                                 (left,right),common,estimate,speeds)
                low, high = extrema(common)
                ll, lu = extrema([(d,sum(ends,Affine())) for d,ends,_ in left])
                rl, ru = extrema([(d,sum(ends,Affine())) for d,ends,_ in right])
                independent = (ll-ru,lu-rl)
                assert independent[0] <= low <= high <= independent[1]
                # Conservative intervals valid for every residual in [lo,hi].
                bounds=[]
                changed=(*state[:6],(state[0]+hi,state[6][1]),*state[7:])
                for candidate_plan in (left_plan,right_plan):
                    bounds.append(exp.interval(changed,candidate_plan,estimate,speeds))
                wl,wu=bounds[0][0]-bounds[1][1],bounds[0][1]-bounds[1][0]
                assert wl <= low <= high <= wu
                stats['pairs'] += 1
                stats['cells'] += len(common)
                stats['paired_certified'] += low>0 or high<0
                stats['independent_certified'] += independent[0]>0 or independent[1]<0
                stats['workload_certified'] += wl>0 or wu<0
                stats['paired_only_vs_independent'] += (low>0 or high<0) and not (independent[0]>0 or independent[1]<0)
                stats['paired_only_vs_workload'] += (low>0 or high<0) and not (wl>0 or wu<0)
                stats['strictly_tighter'] += low>independent[0] or high<independent[1]
                stats['winner_changes'] += low<0<high
                stats['singleton_cells'] += sum(d.lo==d.hi for d,_ in common)
                for domain, delta in common:
                    kind = classify(delta, domain)
                    stats['cell_'+kind] += 1
                    rows.append(dict(seed=args.seed, case=case_index, snapshot=snapshot_index,
                                     pair=stats['pairs'],head=head,
                                     left_candidate=base, right_candidate=alternate,
                                     lo=domain.lo, hi=domain.hi, left=int(domain.left), right=int(domain.right),
                                     intercept=delta.a, slope=delta.b, sign=kind,
                                     pair_lower=low,pair_upper=high,
                                     independent_lower=independent[0],independent_upper=independent[1],
                                     workload_lower=wl,workload_upper=wu))
    print(json.dumps(dict(stats),sort_keys=True))
    if args.summary:
        with args.summary.open('x',encoding='utf-8') as f:
            json.dump(dict(stats),f,indent=2,sort_keys=True)
    return rows


def main(args):
    if args.real:
        rows = real_rows(args)
    else:
        rows = []
        for domain, delta in toy_rows():
            rows.append(dict(seed='toy', snapshot=0, head=0, left_candidate=1, right_candidate=0,
                             lo=domain.lo, hi=domain.hi, left=int(domain.left), right=int(domain.right),
                             intercept=delta.a, slope=delta.b, sign=classify(delta, domain)))
    write_rows(rows, args.out)
    print(f'{len(rows)} common cells written')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--split', choices=('random','witness'), default='random')
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--seed', type=int, default=7680)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--max-pairs', type=int, default=20)
    main(parser.parse_args())


