"""R07 arrival-only replay; recoverable boundaries and executor are unchanged."""
import argparse
from collections import Counter
from copy import deepcopy
from fractions import Fraction as F
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import platform
import sys

from scripts.cpu.dynamic.dynamic_cpu_inputs import (
    FIXTURE, GRID, IO, MANIFEST, MODEL, OVERLAP, ROOT, make_experiment,
    sha256 as file_hash, verify_traces, window_data,
)
from scripts.cpu.replay.study_bailian import Experiment, read, p95
from scripts.cpu.replay.study_boundary_factors import predict
from restore_edge_select.offline_bounds import search

TRACE = 'qwen_traceB_blksz_16.jsonl'
POLICIES = ('local-cost', 'entry', 'entry-replay')
SHORT_VARIANTS = ('original', 'dense', 'sparse', 'even', 'burst')
VARIANTS = (*SHORT_VARIANTS, 'knee-even', 'knee-burst')
CONFIG = ROOT / 'configs/r07-arrival-matrix.json'


def score(row):
    return predict('fifo_snapshot', row)


def transform(trace, variant):
    reqs = trace['requests']
    times = [F(r['arrival_ms']) for r in reqs]
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert times and all(d >= 0 for d in gaps)
    if variant in ('dense', 'sparse'):
        factor = F(1, 2) if variant == 'dense' else F(2)
        gaps = [d * factor for d in gaps]
    if variant.startswith('knee-'):
        gaps = [d * 5 for d in gaps]
        for i in range(0, len(gaps), 8):
            width = min(8, len(gaps) - i)
            total = sum(gaps[i:i + width])
            early = total / (width if variant == 'knee-even' else width * 8)
            gaps[i:i + width] = [early] * (width - 1) + [total - (width - 1) * early]
    if variant in ('even', 'burst'):
        ratio = F(1, 2) if variant == 'even' else F(1, 8)
        for i in range(0, len(gaps) - 1, 2):
            total = gaps[i] + gaps[i + 1]
            gaps[i], gaps[i + 1] = total * ratio, total * (1 - ratio)
    arrival = [times[0]]
    for gap in gaps:
        arrival.append(arrival[-1] + gap)
    if variant in ('original', 'even', 'burst'):
        assert arrival[::2] == times[::2] and arrival[-1] == times[-1]
    if variant.startswith('knee-'):
        assert all(arrival[i] - times[0] == 5 * (times[i] - times[0])
                   for i in {*range(0, len(times), 8), len(times)-1})
    return dict(trace, requests=[dict(r, arrival_ms=str(t)) for r, t in zip(reqs, arrival, strict=True)])


def run(exp, name, *, timed=False, horizon=4, overrides=None, extra=None):
    if name == 'local-cost':
        return exp.run(name, inject_measured=timed, decision_overrides=overrides)
    opts = dict(score_candidate=score, screen_mode='unsettled', queue_horizon=horizon,
                screen_search='entry' if name == 'entry' else 'entry-replay')
    opts.update(extra or {})
    return exp.run('snapshot-screened', **opts, inject_measured=timed, decision_overrides=overrides)


def sequence(result):
    data = [list(result['boundaries_tokens']), [str(v) for v in result['completion_ms']]]
    return sha256(json.dumps(data, separators=(',', ':')).encode()).hexdigest()


def response(exp, result):
    return tuple(t - a for t, a in zip(result['completion_ms'], exp.arrivals, strict=True))


def pack(exp, result, baseline=None):
    r = response(exp, result)
    d = [a-b for a,b in zip(r, baseline, strict=True)] if baseline is not None else []
    # Observable to the evaluation harness, not used by the decision policy.
    return dict(mean=float(result['response_mean_ms']), p95=float(p95(r)), sequence=sequence(result),
                decision=float(sum(row[1] for row in result.get('decision_profile_ns', []))) / 1e6,
                replays=result.get('decision_replays', 0), max_loss=float(max([0, *d])),
                losses_gt10=sum(v > 10 for v in d), late_indices=[i for i,v in enumerate(d) if v > 10],
                mean_loss=float(sum(d) / len(d)) if d else 0.)


def cases(cfg, split):
    fixture, model = read(FIXTURE), read(MODEL)['models'][0]
    if split == 'short':
        for (pattern, times), (rate, budget), variant in product(
            cfg['short_patterns'].items(), cfg['short_strata'], SHORT_VARIANTS
        ):
            spec = deepcopy(fixture)
            spec['cost']['device_copy'] = {'startup_ms': '0', 'bytes_per_ms': 100_000_000}
            spec['cost']['ssd_to_device']['bytes_per_ms'] = str(F(spec['cost']['ssd_to_device']['bytes_per_ms']) / rate)
            a, b, c = spec['requests']
            follow = (b, c, b, c, b, c) if pattern != 'shared' else (b,) * 6
            spec['requests'] = [dict(template, id=str(i), arrival_ms=t)
                                for i,(template,t) in enumerate(zip((a, *follow), times, strict=True))]
            yield dict(window=pattern, rate=rate, budget=budget, variant=variant), Experiment(transform(spec, variant), model, budget)
        return
    directory = ROOT / 'data/raw/bailian'
    verify_traces(directory)
    layouts = [(cfg['layout'], F(1))]
    for window, _, _, trace in window_data(directory, TRACE, cfg[f'{split}_windows'], fixture, model, layouts):
        for (rate,budget),variant in product(cfg['strata'], VARIANTS):
            key = dict(window=window, rate=rate, budget=budget, variant=variant)
            yield key, make_experiment(fixture, model, transform(trace, variant), 'ssd', rate, budget)


def paired(exp, repeats, rotation):
    trials = []
    orders = ((0,1,2), (1,2,0), (2,0,1), (2,1,0), (0,2,1), (1,0,2))
    if repeats:
        for policy in POLICIES:
            run(exp, policy, timed=True)
    for trial in range(repeats):
        order = [POLICIES[i] for i in orders[(trial + rotation) % len(orders)]]
        results = {p:run(exp, p, timed=True) for p in order}
        baseline = response(exp, results['local-cost'])
        trials.append(dict(order=order, policies={p:pack(exp, results[p], baseline) for p in POLICIES}))
    return trials


def visible_decisions(exp, expected):
    seen = {}

    def observe(row):
        # The scorer is called directly by the frozen choose() routine.
        state = sys._getframe(1).f_locals
        seen[state['i']] = (state['arrived'] - state['i'] - 1, state['arrived'] - state['head'])
        return score(row)

    actual = run(exp, 'entry-replay', extra={'score_candidate': observe})
    assert sequence(actual) == sequence(expected)
    return dict(scored=len(seen), beyond_four=sum(a > 4 for a, _ in seen.values()),
                max_followers=max((a for a, _ in seen.values()), default=0),
                max_queue=max((b for _, b in seen.values()), default=0))


def evaluate(cfg, split, out):
    repeats = cfg['timing_repeats'] if split == 'holdout' else 0
    assert not repeats or repeats % 6 == 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8', newline='\n') as stream:
        for idx,(key,exp) in enumerate(cases(cfg,split)):
            z = {p: run(exp,p) for p in POLICIES}
            main = z['entry-replay']
            assert sequence(main) == sequence(z['entry']), key
            assert exp.run('fixed', boundaries=main['boundaries_tokens'])['completion_ms'] == main['completion_ms']
            lo,hi,plan,nodes = search(exp, list(z.values()), cfg['node_budget'], resource_floor=True, dominance=True)
            n = len(exp.arrivals)
            if split == 'short':
                exact = exp.offline()
                optimum = exact['response_sum_ms']
                assert lo <= optimum <= hi
                lo = hi = optimum
                plan = exact['boundaries_tokens']
            assert exp.run('fixed', boundaries=plan)['response_sum_ms'] == hi
            headroom = z['local-cost']['response_sum_ms'] - lo
            widths = [b-a for a,b in zip(exp.arrivals, exp.arrivals[1:])]
            record = dict(key=key, n=n, arrival_span=float(exp.arrivals[-1]-exp.arrivals[0]),
                          arrivals_hash=sha256('|'.join(map(str,exp.arrivals)).encode()).hexdigest(),
                          zero_gaps=sum(g == 0 for g in widths),
                          gap_mean_abs_deviation=float(sum(abs(g-sum(widths)/len(widths)) for g in widths)/len(widths)) if widths else 0.,
                          multi=sum(len(c)>1 for c in exp.eligible), visibility=visible_decisions(exp, main),
                          oracle=dict(lower=float(lo/n), upper=float(hi/n), exact=lo==hi, nodes=nodes,
                                      plan=plan, possible_headroom=float(headroom/n)),
                          zero={p:pack(exp, r, response(exp,z['local-cost'])) for p,r in z.items()},
                          plan=main['boundaries_tokens'], timed=paired(exp,repeats,idx))
            stream.write(json.dumps(record,separators=(',', ':'))+'\n'); stream.flush()
            if key['variant'] == (SHORT_VARIANTS[-1] if split == 'short' else VARIANTS[-1]):
                print(split, idx+1, key['window'], key['rate'], key['budget'], flush=True)
    provenance = dict(python=platform.python_version(), platform=platform.platform(), split=split,
                      files={str(p.relative_to(ROOT)).replace('\\','/'):file_hash(p) for p in (
                          Path(__file__).resolve(), CONFIG, FIXTURE, MODEL, GRID, IO, OVERLAP, MANIFEST,
                          ROOT/'src/restore_edge_select/simulator.py', ROOT/'src/restore_edge_select/offline_bounds.py',
                          ROOT/'scripts/cpu/replay/study_boundary_factors.py',
                          ROOT/'scripts/cpu/replay/study_bailian.py',
                          ROOT/'scripts/cpu/dynamic/dynamic_cpu_inputs.py')}, output_sha256=file_hash(out))
    out.with_suffix('.manifest.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('split', choices=('short','design','holdout'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    cfg = read(CONFIG)
    assert not set(cfg['design_windows']) & set(cfg['holdout_windows'])
    if a.split == 'holdout' and not cfg['locked']:
        p.error('holdout requires the locked R07 matrix')
    evaluate(cfg,a.split,a.out)


if __name__ == '__main__':
    main()
