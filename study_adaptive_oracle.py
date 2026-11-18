"""Audit visible certificates; separately measure regret against a joint offline oracle."""

import argparse
from copy import deepcopy
import csv
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path

from study_bailian import Experiment, ROOT, read
from study_boundary_factors import predict
from restore_edge_select.adaptive import AdaptiveConfig
from restore_edge_select.offline_bounds import search


def check_dominance(fixture, model):
    cases = pruned = 0
    for spacing, budget, shared in product((1, 12, 40), (0, 128, 192), (False, True)):
        trace = deepcopy(fixture)
        trace["cost"]["device_copy"] = {"startup_ms": "0", "bytes_per_ms": 100_000_000}
        originals = trace["requests"][1:]
        trace["requests"] = [trace["requests"][0]]
        for k in range(4):
            request = deepcopy(originals[0 if shared else k % 2])
            trace["requests"].append(dict(request, id=f"R{k}", arrival_ms=1 + k * spacing))
        exp = Experiment(trace, model, budget)
        exact = exp.offline()
        stats = {}
        lo, hi, plan, _ = search(exp, [exp.run("local-cost")], 10000, dominance=True, diagnostics=stats)
        assert lo == hi == exact["response_sum_ms"]
        assert exp.run("fixed", boundaries=plan)["response_sum_ms"] == hi
        cases += 1
        pruned += stats["dominated"]
    assert pruned > 0
    print(f"Checked {cases} five-request exhaustive cases; merged {pruned} dominated states.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resource-floor', action='store_true')
    parser.add_argument('--dominance', action='store_true')
    args = parser.parse_args()
    fixture = read(ROOT / 'fixtures/phase1-hand-trace.json')
    model = read(ROOT / 'configs/qwen3.5-4b.json')['models'][0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', encoding='utf-8', newline='') as stream:
        out = csv.writer(stream, lineterminator='\n')
        out.writerow(('sharing', 'overlap', 'budget', 'ssd_scale', 'c_arrival_ms', 'policy',
                      'mean_ms', 'oracle_mean_ms', 'gap_mean_ms', 'max_loss_vs_local_ms',
                      'decision_replays', 'routes', 'certified', 'largest_certificate_gap_ms', 'boundaries'))
        for shared, overlap, budget, rate, arrival in product((False, True), (False, True),
                (0, 70, 128, 160, 192), (1, 4, 16), (1, 2, 15, 29, 35)):
            trace = deepcopy(fixture)
            if shared:
                trace['requests'][2] = dict(deepcopy(trace['requests'][1]), id='C')
                trace['cost']['device_copy'] = {'startup_ms': '0', 'bytes_per_ms': 100_000_000}
            trace['requests'][2]['arrival_ms'] = arrival
            trace['cost']['ssd_to_device']['bytes_per_ms'] = str(
                F(trace['cost']['ssd_to_device']['bytes_per_ms']) / rate)
            if overlap:
                trace['overlap_slowdown'] = {'ssd_to_device': {'compute': '5/4', 'transfer': '3/2'}}
            exp = Experiment(trace, model, budget)
            oracle = exp.offline()
            if shared and not overlap and budget == 70 and rate == 1 and arrival in (2, 29):
                expected_b = 1152 if arrival == 2 else 576
                assert {plan[1] for plan in oracle['offline_optima']} == {expected_b}
                assert exp.arrivals[1] == 1 < arrival and exp.window >= 1
            local = exp.run('local-cost')
            lo, hi, _, _ = search(exp, [local], max_nodes=1, resource_floor=args.resource_floor,
                                  dominance=args.dominance)
            assert lo <= oracle['response_sum_ms'] <= hi
            lo, hi, plan, _ = search(exp, [local], max_nodes=100, resource_floor=args.resource_floor,
                                     dominance=args.dominance)
            assert lo == hi == oracle['response_sum_ms']
            assert exp.run('fixed', boundaries=plan)['response_sum_ms'] == hi
            opts = dict(screen_mode='unsettled', queue_horizon=4,
                        score_candidate=lambda row: predict('fifo_snapshot', row))
            adaptive = exp.run('adaptive')
            audited = exp.run('adaptive', adaptive=AdaptiveConfig(audit=True))
            assert adaptive['completion_ms'] == audited['completion_ms']
            assert adaptive['boundaries_tokens'] == audited['boundaries_tokens']
            for policy, run in (('local-cost', local), ('longest', exp.run('longest')),
                    ('next4', exp.run('snapshot-screened', **opts)), ('adaptive', adaptive),
                    ('adaptive-strict', exp.run('adaptive', adaptive=AdaptiveConfig(loss_ms=F(0)))),
                    ('offline', oracle)):
                gap = run['response_mean_ms'] - oracle['response_mean_ms']
                assert gap >= 0
                loss = max(0, *(a-b for a,b in zip(run['completion_ms'], local['completion_ms'], strict=True)))
                routes = run.get('adaptive_routes', {})
                out.writerow((int(shared), int(overlap), budget, rate, arrival, policy,
                              float(run['response_mean_ms']), float(oracle['response_mean_ms']), float(gap),
                              float(loss), run.get('decision_replays', 0), sum(r['decisions'] for r in routes.values()),
                              sum(r['certified'] for r in routes.values()),
                              float(max((r['gap_max_ms'] for r in routes.values()), default=0)),
                              json.dumps(run['boundaries_tokens'], separators=(',', ':'))))
    if args.dominance:
        check_dominance(fixture, model)


if __name__ == '__main__':
    main()
