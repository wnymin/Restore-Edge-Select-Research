"""B1 synthetic paired interventions; labels use future outcomes, features do not."""

import argparse
import csv
from bisect import bisect_right
from copy import deepcopy
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path

from scripts.cpu.replay.study_bailian import Experiment, ROOT, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    fixture = read(ROOT / 'fixtures/phase1-hand-trace.json')
    model = read(ROOT / 'configs/qwen3.5-4b.json')['models'][0]
    records = []
    for shared, rate, budget, arrival in product(
            (False, True), (1, 4, 16), (0, 70, 160, 192), (1, 2, 15, 29, 35, 60)):
        trace = deepcopy(fixture)
        trace.pop('expected_by_prefetch_mib', None)
        if shared:
            trace['requests'][2] = dict(deepcopy(trace['requests'][1]), id='C')
        # Hold the copy model fixed across sharing interventions.
        trace['cost']['device_copy'] = {'startup_ms': '0', 'bytes_per_ms': 100_000_000}
        trace['requests'][2]['arrival_ms'] = arrival
        trace['cost']['ssd_to_device']['bytes_per_ms'] = str(
            F(trace['cost']['ssd_to_device']['bytes_per_ms']) / rate)
        exp = Experiment(trace, model, budget)
        base = exp.run('local-cost')
        plan = base['boundaries_tokens']
        snapshots = {}

        def observe(i, now, depth, costs):
            snapshots[i] = dict(decision_ms=now, depth=depth,
                                visible_requests=bisect_right(exp.arrivals, now),
                                candidates=deepcopy(costs))

        oracle = exp.offline()
        for i in (1, 2):
            exp.run('remaining-wait-live', decision_overrides=dict(enumerate(plan[:i])),
                    observe_candidates=observe)
            snapshot = snapshots[i]
            for candidate in trace['requests'][i]['candidates']:
                boundary = candidate['boundary_tokens']
                changed_plan = list(plan)
                changed_plan[i] = boundary
                changed = exp.run('fixed', boundaries=changed_plan)
                delta = [b-a for a, b in zip(base['completion_ms'], changed['completion_ms'], strict=True)]
                gain = base['response_mean_ms'] - changed['response_mean_ms']
                assert sum(delta) == -3 * gain
                assert changed['response_mean_ms'] >= oracle['response_mean_ms']
                if boundary == plan[i]:
                    assert all(d == 0 for d in delta)
                records.append(dict(sharing=shared, ssd_scale=rate, budget_mib=budget,
                    c_arrival_ms=arrival, target=i, baseline_plan=plan, boundary=boundary,
                    snapshot=snapshot, mean_gain_ms=gain, completion_delta_ms=delta,
                    target_delta_ms=delta[i], later_delta_sum_ms=sum(delta[i+1:]),
                    baseline_mean_ms=base['response_mean_ms'],
                    oracle_mean_ms=oracle['response_mean_ms']))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(
        schema=1, seed=None, fixture='fixtures/phase1-hand-trace.json',
        semantics='One fixed boundary change; all other boundaries follow baseline. '
                  'Snapshot is visible before commitment; outcomes and oracle are labels only.',
        records=records), ensure_ascii=False, indent=2,
        default=lambda value: float(value) if isinstance(value, F) else str(value)) + '\n', encoding='utf-8')
    fields = ('sharing', 'ssd_scale', 'budget_mib', 'c_arrival_ms', 'target',
              'boundary', 'baseline_boundary', 'decision_ms', 'visible_requests',
              'prefetch_fits', 'free_reserve_bytes', 'missing_bytes',
              'committed_ahead_compute_ms', 'head_slack_ms', 'inflight_ms',
              'queue_debt_ms', 'mean_gain_ms',
              'target_delta_ms', 'later_delta_sum_ms', 'baseline_mean_ms', 'oracle_mean_ms')
    with args.out.with_suffix('.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for record in records:
            snapshot = record['snapshot']
            row = {**record, **snapshot, **snapshot['candidates'][record['boundary']],
                   'baseline_boundary': record['baseline_plan'][record['target']]}
            writer.writerow({key: float(row[key]) if isinstance(row[key], F) else row[key]
                             for key in fields})
    print(f'{len(records)} interventions written to {args.out}')


if __name__ == '__main__':
    main()
