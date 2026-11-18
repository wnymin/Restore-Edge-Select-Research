"""Two-request exact critical scenarios and a realizable four-corner failure.

No later requests, prefetch competition, active transfer, or persistent cache.
For each feasible target candidate: T = now + max(r, alpha + beta*x) + c.
The native event executor and the general 2-D solver are independent oracles.
"""

import argparse
from fractions import Fraction as F
from itertools import combinations
import json
from pathlib import Path
from random import Random

from scripts.cpu.lookahead.robustness.study_lookahead_joint_domain import Capture, Experiment, ROOT, Region, analyze, oracle, read


def load_lines(exp, record):
    target, fallback, state, base, estimate, speeds = record
    now, head, n, admitted, ready, running, transfer = state[:7]
    assert not exp.per_tier and not exp.dynamic and not exp.host_cache_bytes
    assert exp.release_at == 'start'
    assert (head, target, n) == (0, 1, 2) and len(base) == 2
    assert running is not None and running > now and transfer is None
    assert not admitted and ready == exp.initial and exp.window >= 1
    assert not state[8] and state[9] is None and not state[10]
    assert max(exp.arrivals) <= now
    lines = {}
    for candidate in exp.eligible[target]:
        deps = exp.candidates[target][candidate]
        missing = set(deps)-exp.initial
        assert exp.size(missing) <= exp.budget
        assert not missing.intersection(state[7][:-1])
        alpha = beta = F(0)
        for obj in deps:
            if obj in exp.initial:
                continue
            path = {'RAM': 'ram_to_device', 'SSD': 'ssd_to_device'}[exp.objects[obj]['tier']]
            duration = exp._transfer_times[obj]*estimate.get(path, 1)/speeds[path]
            if path == 'ssd_to_device':
                beta += duration
            else:
                alpha += duration
        suffix = exp._suffix_times[target][candidate]*estimate.get('compute', 1)
        if deps and deps[-1] in exp.initial and candidate < exp.lengths[target]:
            suffix += exp.path_time('device_copy', exp.objects[deps[-1]]['bytes'])*estimate.get('device_copy', 1)
        assert alpha >= 0 and beta >= 0 and suffix >= 0
        lines[candidate] = alpha, beta, suffix
    return lines


def critical_points(lines, box):
    rlo, xlo = box.vertices[0]
    rhi, xhi = box.vertices[2]
    points = set(box.vertices)

    def add(r, x):
        if rlo <= r <= rhi and xlo <= x <= xhi:
            points.add((r, x))

    for alpha, beta, _ in lines.values():
        for x in (xlo, xhi):
            add(alpha+beta*x, x)
        if beta:
            for r in (rlo, rhi):
                add(r, (r-alpha)/beta)
    for (a, b, _), (c, d, _) in combinations(lines.values(), 2):
        if b != d:
            x = (c-a)/(b-d)
            add(a+b*x, x)
    return points


def envelopes(lines, points, fallback):
    upper = {c: None for c in lines}
    worst = {c: None for c in lines}
    regret = {c: F(0) for c in lines}
    target_delta = {c: [] for c in lines}
    for r, x in points:
        values = {c: max(r, a+b*x)+suffix for c, (a, b, suffix) in lines.items()}
        for c in lines:
            delta = values[c]-values[fallback]
            target_delta[c].append(delta)
            upper[c] = max(F(0), delta, upper[c] if upper[c] is not None else F(0))
            worst[c] = delta if worst[c] is None else max(worst[c], delta)
            regret[c] = max(regret[c], values[c]-min(values.values()))
    allowed = [c for c in lines if upper[c] <= 0]
    choose = min(allowed, key=lambda c: (regret[c], c != fallback, -c))
    return dict(individual=upper, total=worst, regret=regret, protected=choose,
                min_target_delta={c: min(v) for c, v in target_delta.items()})


def pairwise_envelopes(lines, box, fallback):
    """At most nine critical scenarios per unordered pair: O(A^2) arithmetic."""
    pair_upper = {(c,c): F(0) for c in lines}
    for a, b in combinations(lines, 2):
        points = critical_points({a: lines[a], b: lines[b]}, box)
        assert len(points) <= 9
        differences = []
        for r, x in points:
            aa, ab, ac = lines[a]
            ba, bb, bc = lines[b]
            differences.append(max(r, aa+ab*x)+ac-max(r, ba+bb*x)-bc)
        pair_upper[a,b] = max(differences)
        pair_upper[b,a] = -min(differences)
    upper = {c: max(F(0), pair_upper[c,fallback]) for c in lines}
    regret = {c: max(pair_upper[c,b] for b in lines) for c in lines}
    safe = [c for c in lines if upper[c] <= 0]
    choice = min(safe, key=lambda c: (regret[c], c != fallback, -c))
    return dict(individual=upper, total={c: pair_upper[c,fallback] for c in lines},
                regret=regret, protected=choice,
                min_target_delta={c: -pair_upper[fallback,c] for c in lines})


def witness_trace():
    trace = read(ROOT/'fixtures/phase1-hand-trace.json')
    trace.pop('expected_by_prefetch_mib', None)
    trace['fixture_id'] = 'two-request-corner-failure'
    trace['purpose'] = 'Synthetic counterexample, not measured hardware costs.'
    trace['requests'] = trace['requests'][:2]
    trace['requests'][0]['arrival_ms'] = 0
    trace['requests'][1]['arrival_ms'] = 53.75
    trace['objects'] = [o for o in trace['objects'] if o['id'].startswith('B.')]
    trace['scheduling']['future_window_requests'] = 1
    trace['scheduling']['initial_device_object_ids'] = ['B.kv1']
    for key in ('a_ms_per_token_rational', 'b_ms_per_pair_rational'):
        trace['cost']['recompute'][key][0] *= 2
    for obj in trace['objects']:
        if obj['id'] == 'B.kv2':
            obj['tier'] = 'SSD'
        elif obj['id'] == 'B.state2':
            obj['tier'] = 'RAM'
    trace['cost']['ram_to_device'] = dict(startup_ms=0, bytes_per_ms=str(F(51511296)/F(51,4)))
    trace['cost']['ssd_to_device'] = dict(startup_ms=0, bytes_per_ms=str(F(51511296)/6))
    return trace


def jsonable(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


class NativeChoice(Experiment):
    """Enact either certificate at the unchanged native first-entry call site."""

    def audit(self, critical):
        self.critical = critical
        self.chosen = None
        self.fallback = None

        def select(rows, fallback, later):
            assert later == 0
            self.fallback = fallback
            return 0

        out = self.run('snapshot-screened', estimate={}, screen_mode='classified',
                       screen_selector=select, score_candidate=lambda row: 0, screen_search='lean')
        return dict(chosen=self.chosen, boundaries=out['boundaries_tokens'],
                    completion=out['completion_ms'])

    def _run(self, policy, threshold_tokens, boundaries, max_combinations, n, **kwargs):
        state = kwargs.get('snapshot')
        if state is None:
            return Experiment._run(self, policy, threshold_tokens, boundaries, max_combinations, n, **kwargs)
        if self.chosen is None:
            record = (1, self.fallback, state, boundaries, kwargs['estimate'], kwargs['forecast_speed'])
            lines = load_lines(self, record)
            box = Region.rectangle(F(6), F(13,2), F(3,4), F(5,4))
            points = critical_points(lines, box) if self.critical else box.vertices
            self.chosen = envelopes(lines, points, self.fallback)['protected']
        # These are comparison values, not fabricated completion measurements.
        return dict(completion_ms=[F(-1) if boundaries[1] == self.chosen else F(0)], events_processed=0)


def build_witness():
    trace = witness_trace()
    model = read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    exp = Capture(trace, model, 256)
    exp.run_trigger({}, 'full')
    assert len(exp.records) == 1
    record = exp.records[0]
    assert record[0] == 1 and record[1] == 576 and record[3][0] == 0
    lines = load_lines(exp, record)
    assert lines == {0: (0, 0, 16), 576: (0, 6, F(17,2)),
                     1152: (F(51,4), F(288,131), 0)}
    box = Region.rectangle(F(6), F(13,2), F(3,4), F(5,4))
    points = critical_points(lines, box)
    full, corners = envelopes(lines, points, 576), envelopes(lines, box.vertices, 576)
    assert pairwise_envelopes(lines, box, 576) == full
    assert full['individual'][1152] == full['total'][1152] == full['regret'][1152] == F(235,524)
    assert corners['individual'][1152] == corners['regret'][1152] == 0
    assert corners['total'][1152] == F(-53,524)
    assert full['protected'] == 576 and corners['protected'] == 1152
    symbolic = analyze(exp, record, rectangle=box)
    for row in symbolic:
        c = row['candidate']
        assert F(row['individual_upper']) == full['individual'][c]
        assert F(row['total_upper']) == full['total'][c]
        assert F(row['regret']) == full['regret'][c]
    checks = []
    nominal = F(25,4), F(1)
    for point in sorted(points | {nominal}):
        vectors = {}
        for c in lines:
            plan = (0, c)
            vector = oracle(exp, record[2], plan, {}, record[5], point)
            a, b, suffix = lines[c]
            assert vector == [record[2][0]+point[0], record[2][0]+max(point[0], a+b*point[1])+suffix]
            vectors[c] = vector
        checks.append(dict(point=point, completion=vectors,
                           deep_minus_shallow=vectors[1152][1]-vectors[576][1]))
    actual = {c: Experiment._run(exp, 'fixed', 0, (0,c), 100, 2)['completion_ms'] for c in lines}
    assert actual[576] == [F(60), F(137,2)]
    assert actual[1152][1]-actual[576][1] == F(26,131)
    assert next(p for p in checks if p['point'] == nominal)['completion'] == actual
    corner_run = NativeChoice(trace, model, 256).audit(False)
    critical_run = NativeChoice(trace, model, 256).audit(True)
    assert corner_run == dict(chosen=1152, boundaries=[0,1152], completion=actual[1152])
    assert critical_run == dict(chosen=576, boundaries=[0,576], completion=actual[576])
    return jsonable(dict(trace=trace, budget_mib=256, fallback=576,
                         domain=dict(r=['6','13/2'], x=['3/4','5/4']), lines=lines,
                         critical_points=sorted(points), full=full, corners=corners,
                         symbolic_rows=symbolic, numeric_checks=checks,
                         native_actual_completion=actual, native_actual_delta=F(26,131),
                         native_corner_policy=corner_run, native_critical_policy=critical_run))


def stress_checks(seed, rounds):
    """Algebraic model checks; these are not additional native trace samples."""
    rng = Random(seed)
    failures = 0
    common_slope_checks = 0
    for _ in range(rounds):
        count = rng.randrange(2, 9)
        box = Region.rectangle(F(rng.randrange(1,10)), F(rng.randrange(10,21)), F(1,4), F(7,4))
        lines = {c: (F(rng.randrange(21)), F(rng.randrange(16)), F(rng.randrange(16)))
                 for c in range(count)}
        points = critical_points(lines, box)
        exact, corner = envelopes(lines, points, 0), envelopes(lines, box.vertices, 0)
        assert pairwise_envelopes(lines, box, 0) == exact
        assert all(corner['individual'][c] <= exact['individual'][c] and
                   corner['regret'][c] <= exact['regret'][c] for c in lines)
        failures += any(corner['individual'][c] < exact['individual'][c] for c in lines)
        # Equal SSD coefficients imply a common coordinate z=r-beta*x.
        beta = F(rng.randrange(16))
        equal = {c: (a, beta, suffix) for c, (a, _, suffix) in lines.items()}
        exact = envelopes(equal, critical_points(equal, box), 0)
        corner = envelopes(equal, box.vertices, 0)
        two = envelopes(equal, (box.vertices[1], box.vertices[3]), 0)
        assert exact == corner == two
        # A no-load candidate T=r+c also depends on the same z after subtracting beta*x.
        equal[0] = (F(0), F(0), equal[0][2])
        exact = envelopes(equal, critical_points(equal, box), 0)
        two = envelopes(equal, (box.vertices[1], box.vertices[3]), 0)
        assert exact == two
        common_slope_checks += count
    return dict(seed=seed, algebraic_cases=rounds, unequal_slope_cases_with_corner_gap=failures,
                equal_slope_candidate_checks=common_slope_checks)


def native_stress_checks(seed=7781, rounds=64):
    """Legal cache objects and original-executor vectors, unlike algebraic stress."""
    rng = Random(seed)
    model = read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    cases = []
    for index in range(rounds):
        trace = witness_trace()
        trace['fixture_id'] = 'two-request-projection-audit'
        trace['requests'][1]['arrival_ms'] = 0
        factor = rng.choice((F(1,4), F(1,2), F(1), F(2), F(4)))
        for key in ('a_ms_per_token_rational', 'b_ms_per_pair_rational'):
            a, b = trace['cost']['recompute'][key]
            scaled = F(a,b)*factor
            trace['cost']['recompute'][key] = [scaled.numerator, scaled.denominator]
        for obj in trace['objects']:
            obj['tier'] = rng.choice(('RAM','SSD'))
        trace['scheduling']['initial_device_object_ids'] = [o['id'] for o in trace['objects'] if rng.randrange(4)==0]
        trace['cost']['device_copy'] = dict(startup_ms='1/10', bytes_per_ms=str(1<<20))
        for path in ('ram_to_device', 'ssd_to_device'):
            trace['cost'][path] = dict(startup_ms=str(rng.randrange(4)),
                                       bytes_per_ms=str(F(16<<20,rng.randrange(1,33))))
        exp = Capture(trace, model, 256)
        exp.run_trigger({}, 'full')
        assert len(exp.records) == 1
        record = exp.records[0]
        lines = load_lines(exp, record)
        residual = record[2][5]-record[2][0]
        box = Region.rectangle(residual/4, residual*F(7,4), F(1,4), F(7,4))
        points = critical_points(lines, box)
        exact = pairwise_envelopes(lines, box, record[1])
        assert exact == envelopes(lines, points, record[1])
        full = analyze(exp, record, rectangle=box)
        for row in full:
            c = row['candidate']
            assert F(row['individual_upper']) == exact['individual'][c]
            assert F(row['total_upper']) == exact['total'][c]
            assert F(row['regret']) == exact['regret'][c]
            assert row['protected'] == exact['protected']
        for point in points:
            for c, (alpha,beta,suffix) in lines.items():
                actual = oracle(exp, record[2], (0,c), {}, record[5], point)
                assert actual == [record[2][0]+point[0],
                                  record[2][0]+max(point[0],alpha+beta*point[1])+suffix]
        cases.append(dict(case=index, fallback=record[1], lines=lines,
                          initial=sorted(exp.initial), tiers={o['id']:o['tier'] for o in trace['objects']},
                          critical_points=len(points), critical_vectors=len(points)*len(lines),
                          symbolic_cells=full[0]['cells'], symbolic_vectors=full[0]['probes'],
                          exact=exact))
    return jsonable(dict(seed=seed, native_cases=rounds, cases=cases,
                         critical_vectors=sum(c['critical_vectors'] for c in cases),
                         symbolic_vectors=sum(c['symbolic_vectors'] for c in cases)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    parser.add_argument('--rounds', type=int, default=1000)
    parser.add_argument('--native-out', type=Path)
    parser.add_argument('--native-check', type=Path)
    args = parser.parse_args()
    payload = build_witness()
    payload['algebraic_stress'] = stress_checks(7780, args.rounds)
    if args.check:
        assert payload == read(args.check)
        print('Native counterexample, symbolic certificate and algebraic checks reproduced.')
    if args.out:
        with args.out.open('x', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    if args.native_out or args.native_check:
        native = native_stress_checks()
        if args.native_check:
            assert native == read(args.native_check)
        if args.native_out:
            with args.native_out.open('x', encoding='utf-8') as f:
                json.dump(native, f, ensure_ascii=False, indent=2)
        print('Native projection stress:', {k:v for k,v in native.items() if k!='cases'})
    print(json.dumps(dict(native_actual_delta=payload['native_actual_delta'],
                          critical_points=len(payload['critical_points']),
                          stress=payload['algebraic_stress']), sort_keys=True))
