"""Check R04 masking and queue debt against every legal fixed boundary."""
import json
from copy import deepcopy
from fractions import Fraction as F

from scripts.cpu.replay.study_bailian import Experiment, ROOT, read
from restore_edge_select.adaptive import AdaptiveConfig, projected_priority, select


def snapshot(exp, previous):
    captured = []

    def capture(rows, base, followers):
        assert not followers
        captured.append((rows, base))

    exp.run('snapshot-screened', score_candidate=lambda _: 0,
            screen_mode='classified', screen_selector=capture,
            decision_overrides=dict(enumerate(previous)))
    rows, base = captured.pop()
    assert not captured
    vectors = {c: exp.run('fixed', boundaries=[*previous, c])['completion_ms'] for c in rows}
    return rows, base, vectors


def measure(exp, previous, *, factor):
    rows, base, vectors = snapshot(exp, previous)
    prefix_compute = [exp.recompute(i, c) for i, c in enumerate(previous)]
    floors = {c: (*prefix_compute[:1], *[0] * (len(previous) - 1),
                   sum(prefix_compute) + exp.recompute(len(previous), c)) for c in rows}
    exact = min(rows, key=lambda c: (sum(vectors[c]), -c))
    result = {'oracle': exact, 'completion_ms': vectors, 'base': base,
              'coverage_ms': rows[base]['head_slack_ms'] + rows[base]['committed_ahead_compute_ms'],
              'queue_debt_ms': rows[base]['queue_debt_ms'], 'inflight_ms': rows[base]['inflight_ms'],
              'current_admission': {c: r['prefetch_fits'] for c, r in rows.items()},
              'visible_requests': len(previous) + 1, 'factor': factor}
    for name, options in [('full', {}), ('ablated', {factor: False})]:
        scores = {c: projected_priority(r, **options) for c, r in rows.items()}
        cfg = AdaptiveConfig(light_replays=2, coupled_replays=2, loss_ms=None,
                             gain_per_request_ms=F(0), audit=True)
        action, _, gap, _, _ = select(rows, base, exp.budget, cfg, floors, vectors.__getitem__, scores)
        exhaustive = AdaptiveConfig(light_replays=len(rows), coupled_replays=len(rows),
                                    loss_ms=None, gain_per_request_ms=F(0), audit=True)
        exact_action, _, exact_gap, _, _ = select(rows, base, exp.budget, exhaustive, floors,
                                                 vectors.__getitem__, scores)
        assert exact_gap == 0 and sum(vectors[exact_action]) == sum(vectors[exact])
        regret = sum(vectors[action]) - sum(vectors[exact])
        assert 0 <= regret <= gap
        result[name] = {'scores': scores, 'priority_winner': min(rows, key=lambda c: (scores[c], -c)),
                        'action': action, 'regret_ms': regret, 'gap_ms': gap}
    return result


def main():
    fixture = read(ROOT / 'fixtures/phase1-hand-trace.json')
    model = read(ROOT / 'configs/qwen3.5-4b.json')['models'][0]
    coverage = []
    for budget in (70, 128):
        for arrival in ('26', '107/4', '27', '30'):
            trace = deepcopy(fixture)
            trace['requests'] = trace['requests'][:2]
            trace['requests'][1]['arrival_ms'] = arrival
            row = measure(Experiment(trace, model, budget), [0], factor='include_coverage')
            coverage.append(dict(budget_mib=budget, arrival_ms=arrival, **row))
    assert [r['oracle'] for r in coverage[:4]] == [576, 1152, 1152, 1152]
    assert coverage[1]['completion_ms'][576] == coverage[1]['completion_ms'][1152]
    assert all(r['full']['priority_winner'] == r['oracle'] for r in coverage)
    for row in coverage:
        assert row['inflight_ms'] == row['queue_debt_ms'] == 0
        for boundary, (head, target) in row['completion_ms'].items():
            assert head == 30 and target == head + row['full']['scores'][boundary]
    assert coverage[0]['ablated']['priority_winner'] != coverage[0]['oracle']

    queue_debt = []
    for budget in (70, 128, 160, 192):
        for preceding_boundary in (576, 1152):
            for arrival in ('1', '3/2', '2', '3', '4', '5'):
                trace = deepcopy(fixture)
                trace['requests'][2]['arrival_ms'] = arrival
                row = measure(Experiment(trace, model, budget), [0, preceding_boundary],
                              factor='include_queue_debt')
                queue_debt.append(dict(budget_mib=budget, arrival_ms=arrival,
                                       preceding_boundary=preceding_boundary, **row))
    # Shorten the running head while keeping the same-time B/C admission.
    # B's transfer is queued and has not started when C chooses.
    for head_tokens in (576, 1152):
        for budget in (128, 192):
            for preceding_boundary in (576, 1152):
                trace = deepcopy(fixture)
                trace['requests'][0]['input_token_runs'][0]['count'] = head_tokens
                row = measure(Experiment(trace, model, budget), [0, preceding_boundary],
                              factor='include_queue_debt')
                queue_debt.append(dict(budget_mib=budget, arrival_ms='1', head_tokens=head_tokens,
                                       preceding_boundary=preceding_boundary, **row))
    assert any(r['queue_debt_ms'] > 0 for r in queue_debt)
    assert any(r['full']['scores'] != r['ablated']['scores'] for r in queue_debt)
    target = ROOT / 'docs/evidence/r04-fixed-snapshots.json'
    target.write_text(json.dumps({'coverage': coverage, 'queue_debt': queue_debt},
                                 indent=2, default=str) + '\n', encoding='utf-8')
    print(f'Checked {len(coverage)} coverage and {len(queue_debt)} queue-debt snapshots')


if __name__ == '__main__':
    main()
