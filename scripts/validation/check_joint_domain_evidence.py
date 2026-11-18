"""Reconstruct joint-domain statistics and replay against the original executor."""

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction as F
import json
from pathlib import Path

from scripts.validation.lookahead_evidence import csv_path, csv_paths, summary_path
from scripts.cpu.lookahead.robustness.study_lookahead_joint_domain import (
    Capture, Experiment, ROOT, Region, analyze, constructed_check, geometry_checks,
    oracle, partition, random_cases, read,
)


def inside(region, point):
    """Independent closed-hull test followed by strict-face exclusions."""
    vertices = region.vertices
    dimension = region.dimension()
    if dimension == 0:
        closed = point == vertices[0]
    elif dimension == 1:
        a, b = vertices[:2]
        closed = ((b[0]-a[0])*(point[1]-a[1]) == (b[1]-a[1])*(point[0]-a[0]) and
                  all(min(p[i] for p in vertices) <= point[i] <= max(p[i] for p in vertices)
                      for i in range(2)))
    else:
        closed = all((b[0]-a[0])*(point[1]-a[1]) >= (b[1]-a[1])*(point[0]-a[0])
                     for a, b in zip(vertices, vertices[1:]+vertices[:1]))
    return closed and all(d.at(point) < 0 for d in region.strict)


def check_cover(exp, record, radius):
    target, fallback, state, base, estimate, speeds = record
    residual = state[5]-state[0]
    box = Region.rectangle(residual*(1-radius), residual*(1+radius), 1-radius, 1+radius)
    plans = {c: tuple(c if j == target else b for j, b in enumerate(base))
             for c in exp.eligible[target]}
    cells = partition(exp, state, plans, estimate, speeds, box)
    points = {p for region, _ in cells for p in region.vertices}
    rlo, xlo = box.vertices[0]
    rhi, xhi = box.vertices[2]
    points |= {(rlo+(rhi-rlo)*F(i,4), xlo+(xhi-xlo)*F(j,4))
               for i in range(5) for j in range(5)}
    for point in points:
        matches = [(region, vectors) for region, vectors in cells if inside(region, point)]
        assert len(matches) == 1, ('coverage/overlap', point, len(matches))
        for c, vector in matches[0][1].items():
            actual = oracle(exp, state, plans[c], estimate, speeds, point)
            assert actual == [a.at(point) for a in vector]
    return len(points)*len(plans)


def check(path):
    rows = list(csv.DictReader(path.open(encoding='utf-8')))
    groups = defaultdict(list)
    for row in rows:
        groups[int(row['decision'])].append(row)
    stats = Counter()
    for decision, group in groups.items():
        first = group[0]
        assert decision >= 1 and len(group) == len({r['candidate'] for r in group})
        stats['decisions'] += 1
        stats['candidates'] += len(group)
        for name in ('cells', 'faces', 'edges', 'vertices', 'probes', 'inflight',
                     'sensitive_r', 'sensitive_x'):
            assert all(row[name] == first[name] for row in group)
            stats[name] += int(first[name])
        assert int(first['cells']) == sum(int(first[k]) for k in ('faces', 'edges', 'vertices'))
        stats['both_sensitive'] += int(first['sensitive_r']) and int(first['sensitive_x'])
        stats['constraint_changes_action'] += first['protected'] != first['robust']
        fallback = int(first['fallback'])
        by_candidate = {int(r['candidate']): r for r in group}
        assert F(by_candidate[fallback]['individual_upper']) == 0
        assert F(by_candidate[fallback]['total_upper']) == 0
        safe = []
        for row in group:
            candidate = int(row['candidate'])
            h, corner = F(row['individual_upper']), F(row['corner_upper'])
            assert corner <= h and F(row['sampled_max']) <= h
            assert int(row['safe']) == int(h <= 0)
            assert int(row['corners_false_safe']) == int(corner <= 0 < h)
            assert F(row['regret']) >= 0
            if h <= 0:
                safe.append(candidate)
            stats['corners_false_safe'] += int(row['corners_false_safe'])
        key = lambda c: (F(by_candidate[c]['regret']), c != fallback, -c)
        assert int(first['robust']) == min(by_candidate, key=key)
        assert int(first['protected']) == min(safe, key=key)
    summary = read(summary_path(path))
    for key, value in stats.items():
        assert summary[key] == value, (path.name, key, summary[key], value)
    assert summary['configurations'] >= len({r['case'] for r in rows})
    print(path.name, json.dumps(dict(stats), sort_keys=True))
    return rows


def replay(path, rounds, radius, native_arrivals):
    archived = list(csv.DictReader(path.open(encoding='utf-8')))
    seed = int(archived[0]['seed'])
    by_key = {(int(r['case']), int(r['target']), int(r['candidate'])): r for r in archived}
    model = read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    decisions, probes = 0, 0
    for case, (trace, labels, estimate) in enumerate(random_cases(model, seed, rounds)):
        if not native_arrivals:
            for request in trace['requests']:
                request['arrival_ms'] = 0
        exp = Capture(trace, model, labels['budget'])
        exp.run_trigger(estimate, 'full')
        for record in exp.records:
            state = record[2]
            if state[5] is None or state[5] <= state[0]:
                continue
            rows = analyze(exp, record, radius)
            for row in rows:
                previous = by_key[case, row['target'], row['candidate']]
                assert all(str(value) == previous[key] for key, value in row.items())
            probes += check_cover(exp, record, radius)
            decisions += 1
    print('replayed', decisions, 'decisions; independently checked', probes, 'grid/stratum vectors')


def corner_audit(path, radius, native_arrivals):
    """Independently compare four actual scenarios to exact domain envelopes."""
    archived = list(csv.DictReader(path.open(encoding='utf-8')))
    by_decision = defaultdict(list)
    for row in archived:
        by_decision[int(row['case']), int(row['target'])].append(row)
    seed = int(archived[0]['seed'])
    rounds = read(summary_path(path))['configurations']
    model = read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    for case, (trace, labels, estimate) in enumerate(random_cases(model, seed, rounds)):
        if not native_arrivals:
            for request in trace['requests']:
                request['arrival_ms'] = 0
        exp = Capture(trace, model, labels['budget'])
        exp.run_trigger(estimate, 'full')
        for record in exp.records:
            target, fallback, state, base, costs, speeds = record
            if (case, target) not in by_decision:
                continue
            group = by_decision[case, target]
            first = group[0]
            box = Region.rectangle(F(first['rlo']), F(first['rhi']), F(first['xlo']), F(first['xhi']))
            assert box.vertices[0] == ((state[5]-state[0])*(1-radius), 1-radius)
            plans = {c: tuple(c if j == target else b for j, b in enumerate(base))
                     for c in exp.eligible[target]}
            regrets = {c: F(0) for c in plans}
            uppers, totals = {c: None for c in plans}, {c: None for c in plans}
            for point in box.vertices:
                vectors = {c: oracle(exp, state, p, costs, speeds, point) for c, p in plans.items()}
                scores = {c: sum(v) for c, v in vectors.items()}
                for c in plans:
                    h = max(a-b for a, b in zip(vectors[c], vectors[fallback]))
                    t = scores[c]-scores[fallback]
                    uppers[c] = h if uppers[c] is None else max(uppers[c], h)
                    totals[c] = t if totals[c] is None else max(totals[c], t)
                    regrets[c] = max(regrets[c], scores[c]-min(scores.values()))
            safe = [c for c in plans if uppers[c] <= 0]
            corner_choice = min(safe, key=lambda c: (regrets[c], c != fallback, -c))
            for row in group:
                c = int(row['candidate'])
                assert uppers[c] == F(row['corner_upper'])
                assert regrets[c] <= F(row['regret']) and totals[c] <= F(row['total_upper'])
                yield dict(seed=seed, case=case, target=target, candidate=c,
                           individual_gap=str(F(row['individual_upper'])-uppers[c]),
                           total_gap=str(F(row['total_upper'])-totals[c]),
                           regret_gap=str(F(row['regret'])-regrets[c]),
                           corner_regret=str(regrets[c]), full_regret=row['regret'],
                           corner_choice=corner_choice, full_choice=row['protected'],
                           changed=int(corner_choice != int(row['protected'])))


def all_corners():
    settings = [('7770', F(1,4), False), ('7771-wide', F(3,4), False),
                ('7772-arrivals', F(3,4), True)]
    return [row for suffix, radius, native in settings
            for row in corner_audit(csv_path(f'dynamic-lookahead-joint-domain-{suffix}.csv'),
                                    radius, native)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--radius', default='1/4')
    parser.add_argument('--native-arrivals', action='store_true')
    parser.add_argument('--corner-out', type=Path)
    args = parser.parse_args()
    geometry_checks()
    constructed = constructed_check()
    saved = ROOT/'docs/evidence/dynamic-lookahead-joint-domain-constructed.json'
    if saved.exists():
        assert constructed == read(saved)['rows']
    for path in csv_paths('dynamic-lookahead-joint-domain-*.csv'):
        if not path.stem.endswith('-corners'):
            check(path)
    corner_path = csv_path('dynamic-lookahead-joint-domain-corners.csv')
    actual = list(csv.DictReader(corner_path.open(encoding='utf-8')))
    expected = [{k: str(v) for k, v in row.items()} for row in all_corners()]
    assert actual == expected
    print('reproduced all', len(expected), 'corner comparisons against original executor')
    if args.replay:
        replay(args.replay, args.rounds, F(args.radius), args.native_arrivals)
    if args.corner_out:
        rows = all_corners()
        with args.corner_out.open('x', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print('corner audit', len(rows), 'candidates;',
              {k: sum(F(r[k]) > 0 for r in rows) for k in ('individual_gap','total_gap','regret_gap')},
              'changed candidates', sum(r['changed'] for r in rows))
