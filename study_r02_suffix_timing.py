"""Paired CPU timing of complete replay versus paused suffix reuse."""
import argparse
from copy import deepcopy
from fractions import Fraction as F
from itertools import permutations
import json
from pathlib import Path
from random import Random
from statistics import median
from time import perf_counter_ns

from scripts.cpu.lookahead.screening.study_lookahead_coalescence import Coalescence
from scripts.cpu.lookahead.screening.study_lookahead_workload_bounds import clone
from scripts.cpu.lookahead.staging.study_r02_suffix_keys import Experiment, ROOT, read


def trial(exp, initial, mode):
    cache, vectors = {}, []
    events = hits = 0
    speeds = dict(ram_to_device=F(1), ssd_to_device=F(1))
    for boundary in (0, 576, 1152):
        plan = (0, boundary, 1152, 576)

        def run(state, stop=None):
            return Experiment._run(exp, 'fixed', 0, plan, 100, 4,
                                   snapshot=clone(state), forecast_speed=speeds,
                                   lean_rollout=True, pause_at_head=stop)

        if mode == 'reference':
            result = run(initial)
            vectors.append(result['completion_ms'])
            events += result['events_processed']
            continue
        part = run(initial, 2)
        events += part['events_processed']
        state = part['snapshot']
        now = state[0]
        key = exp.signature(state, plan)
        if key in cache:
            hits += 1
            suffix = [now + offset for offset in cache[key]]
        else:
            result = run(state)
            events += result['events_processed']
            suffix = result['completion_ms']
            cache[key] = tuple(t - now for t in suffix)
        vectors.append(part['completion_ms'] + suffix)
    return vectors, events, hits


def measure(split, repeats):
    rng = Random(72843)
    modes = ('reference', 'full', 'projected')
    model = read(ROOT / 'configs/qwen3.5-4b.json')['models'][0]
    frozen = read(ROOT / 'docs/evidence/r02-cpu-inputs.json')
    rows = []
    for case in frozen['cases']:
        if case['split'] != split:
            continue
        trace = deepcopy(case['trace'])
        for request in trace['requests']:
            request['arrival_ms'] = 0
        exp = Coalescence(trace, model, F(case['capacity_bytes'], 1 << 20))
        exp.future_objects = {2: set(exp.candidates[2][1152]) | set(exp.candidates[3][576])}
        initial = (F(0), 0, 4, {}, set(exp.initial), None, None, (), set(), None, {})
        expected = trial(exp, initial, 'reference')[0]
        orders = list(permutations(modes))
        for repeat in range(repeats):
            rng.shuffle(orders)
            for order in orders:
                for position, mode in enumerate(order):
                    exp.project = mode == 'projected'
                    t = perf_counter_ns()
                    vectors, events, hits = trial(exp, initial, mode)
                    elapsed = perf_counter_ns() - t
                    assert vectors == expected, (case['id'], mode)
                    rows.append(dict(case=case['id'], mode=mode, round=repeat,
                                     order=order, position=position, ns=elapsed,
                                     events=events, hits=hits))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('design', 'holdout'), default='holdout')
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = measure(args.split, args.repeats)
    args.out.write_text(json.dumps(rows, separators=(',', ':')) + '\n', encoding='utf-8')
    for mode in ('reference', 'full', 'projected'):
        group = [row for row in rows if row['mode'] == mode]
        durations = sorted(row['ns'] for row in group)
        print(mode, 'median_ns', median(durations), 'p95_ns', durations[(95 * len(durations) + 99)//100-1],
              'events', sum(row['events'] for row in group), 'hits', sum(row['hits'] for row in group))
