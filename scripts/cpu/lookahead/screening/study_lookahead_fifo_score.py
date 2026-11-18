"""Measure an exact score-only FIFO suffix reduction in Phase 1."""

import argparse
from collections import Counter
import csv
from fractions import Fraction as F
from itertools import islice, product
import json
from pathlib import Path
from time import perf_counter_ns

from scripts.cpu.lookahead.screening.study_lookahead_phase1 import (Experiment, ROOT, SerialTailExperiment, inputs,
                                    opts, p95, read, run_policy, trace_for, without_cross_request_sharing)
from scripts.cpu.replay.study_bailian import build


class ActiveTailScore(SerialTailExperiment):
    """Private adapter for the full-visible, sum-only scored forecast call site."""

    def run_score(self, estimate, *, measure=False):
        self.distance = None
        self.cut = None
        self.offset = None
        self.stats.clear()

        def begin(observable, _fallback, later):
            self.distance = next(iter(observable.values()))['head_distance']
            self.cut = None
            self.offset = None
            return 0

        start = perf_counter_ns()
        result = self.run('snapshot-screened', **(opts(0, estimate) | {'screen_selector': begin}),
                          inject_measured=measure)
        return result, (perf_counter_ns() - start) / 1e6

    def _run(self, policy, threshold_tokens, boundaries, max_combinations, n, **kwargs):
        state = kwargs.get('snapshot')
        if state is None:
            return Experiment._run(self, policy, threshold_tokens, boundaries, max_combinations, n, **kwargs)
        assert policy == 'fixed' and kwargs.get('lean_rollout')
        assert kwargs.get('future_cost') is None and kwargs.get('completion_cutoff') is None
        assert kwargs.get('prefetch_budget') in (None, self.budget)
        assert kwargs.get('observe_candidates') is None and self.distance is not None
        now, head, arrived, admitted = state[:4]
        target = head + self.distance
        assert head <= target < n == arrived and self.arrivals[n-1] <= now
        if self.audit:
            # The simulator consumes its private snapshot containers in place.
            checked_state = (*state[:3], dict(admitted), set(state[4]), *state[5:8],
                             set(state[8]), state[9], dict(state[10]))
        if self.cut is None:
            cut = n
            while cut > target + 1:
                self.stats['suffix_metadata_checks'] += 1
                if cut - 1 in admitted or self.can_prefetch[cut-1][boundaries[cut-1]]:
                    break
                cut -= 1
            self.cut = cut
            self.stats['decisions'] += 1
            self.stats['reduced_decisions'] += cut < n
            if self.audit:
                self.signature = (checked_state, boundaries[:target], boundaries[target+1:])
        elif self.audit:
            assert self.signature == (checked_state, boundaries[:target], boundaries[target+1:])
        self.stats['forecasts'] += 1
        cut = self.cut
        prefix = Experiment._run(self, policy, threshold_tokens, boundaries[:cut], max_combinations, cut,
                                 **(kwargs | {'snapshot': (*state[:2], cut, *state[3:])}))
        terms = prefix['completion_ms']
        if cut < n:
            self.stats['reduced_forecasts'] += 1
            self.stats['avoided_request_expansions'] += n - cut
            assert terms[-1] >= now
            # These are score terms, not a vector of actual timestamps. The
            # caller only sums them; the omitted work term is common to all arms.
            terms.append((n - cut) * terms[-1])
        if self.audit and cut < n:
            # Do not reuse the containers saved as the decision signature.
            full_state = (*checked_state[:3], dict(checked_state[3]), set(checked_state[4]), *checked_state[5:8],
                          set(checked_state[8]), checked_state[9], dict(checked_state[10]))
            full = Experiment._run(self, policy, threshold_tokens, boundaries, max_combinations, n,
                                   **(kwargs | {'snapshot': full_state}))
            shift = sum(full['completion_ms']) - sum(terms)
            if self.offset is None:
                self.offset = shift
            else:
                assert shift == self.offset
            self.stats['audited'] += 1
        return prefix


def measure_case(trace, model, capacity, estimate, labels, timed, *, joint=False, audit=False):
    exp = Experiment(trace, model, capacity)
    oracle = exp.offline() if joint else None
    results = {}
    for name in ('target', 'fixed4', 'all', 'compact'):
        if name == 'compact':
            if audit:
                checked = ActiveTailScore(trace, model, capacity, audit=True)
                test, _ = checked.run_score(estimate)
                assert test['boundaries_tokens'] == results['all']['boundaries_tokens']
                assert test['completion_ms'] == results['all']['completion_ms']
                audit_count = checked.stats['audited']
            else:
                audit_count = 0
            compact = ActiveTailScore(trace, model, capacity)
            out, wall = compact.run_score(estimate)
            stats = compact.stats.copy()
            assert out['boundaries_tokens'] == results['all']['boundaries_tokens']
            assert out['completion_ms'] == results['all']['completion_ms']
            actual, _ = compact.run_score(estimate, measure=True) if timed else (None, None)
        else:
            out, elapsed = run_policy(exp, name, estimate)
            wall = elapsed / 1e6
            stats, audit_count = Counter(), 0
            actual, _ = run_policy(exp, name, estimate, timed=True) if timed else (None, None)
        results[name] = out
        base = results['target']
        delta = [a-b for a,b in zip(out['completion_ms'], base['completion_ms'], strict=True)]
        times = [a-b for a,b in zip(out['completion_ms'], exp.arrivals, strict=True)]
        yield dict(**labels, policy=name, requests=len(times),
                   boundary_changes=sum(a != b for a,b in zip(out['boundaries_tokens'], base['boundaries_tokens'], strict=True)),
                   response_gain_ms=float(-sum(delta)), worst_vs_target_ms=float(max(0,*delta)),
                   p95_ms=float(p95(times)), response_sum_ms=float(out['response_sum_ms']),
                   regret_joint_ms=float(out['response_sum_ms']-oracle['response_sum_ms']) if joint else '',
                   oracle_combinations=oracle['combinations_evaluated'] if joint else '',
                   replays=out['decision_replays'], events=out['decision_events'], audited=audit_count,
                   reduced_decisions=stats['reduced_decisions'], reduced_forecasts=stats['reduced_forecasts'],
                   checked_suffix=stats['suffix_metadata_checks'], saved_request_expansions=stats['avoided_request_expansions'],
                   wall_ms=wall, timed_sum_ms=float(actual['response_sum_ms']) if timed else '',
                   decision_cpu_ms=sum(t for _,t in actual['decision_profile_ns'])/1e6 if timed else '')


def witness(model, args):
    # Cross the known admission soft spot from the exploratory fixture sweep,
    # and retain both capacity and sharing controls, including all zero gains.
    for mode, capacity, suffix, bias in product(('distinct','alias'), (248, 256), (0, 4, 16, 64), (F(3,4), F(1), F(5,4))):
        trace = trace_for(mode, model, free_case=True)
        for r in trace['requests'][1:]:
            r['arrival_ms'] = 1
        trace['cost']['ssd_to_device']['bytes_per_ms'] = str(F(trace['cost']['ssd_to_device']['bytes_per_ms'])*2)
        for j in range(suffix):
            trace['requests'].append(dict(id=f'T{j}', arrival_ms=1,
                                          input_token_runs=[dict(token_id=10000+j, count=32)],
                                          candidates=[dict(boundary_tokens=0, load_order=[])]))
        labels = dict(split='witness', source=mode, window='', sharing=mode, ssd_scale='1/2',
                      estimate=str(bias), budget=capacity, suffix=suffix)
        yield from measure_case(trace, model, capacity, {'ssd_to_device':bias}, labels, args.timed,
                                joint=True, audit=True)


def independent_windows(fixture, model):
    # Fixed from available source names/lengths, before looking at any replay
    # outcomes. Neither prior raw windows nor stress multipliers are reused.
    sources = ('qwen_coder_blksz_16.jsonl', 'qwen_traceA_blksz_16.jsonl',
               'qwen_traceB_blksz_16.jsonl', 'qwen_thinking_blksz_16.jsonl')
    for i, source in enumerate(sources):
        window = 11 + 2 * i
        with (ROOT/'data/raw/bailian'/source).open(encoding='utf-8') as f:
            rows = [json.loads(line, parse_float=str) for line in islice(f, 352*window, 352*(window+1))]
        assert len(rows) == 352
        for spacing in (F(1), F(1, 32)):
            trace = build(rows, 256, fixture, model, spacing, 'split', 3584)
            for sharing in ('native','disabled'):
                yield source, window, sharing, spacing, trace if sharing == 'native' else without_cross_request_sharing(trace)


def public(model, args):
    fixture = read(ROOT/'fixtures/phase1-hand-trace.json')
    grid = read(ROOT/'docs/evidence/prefill-vulkan-4k.json')
    paths = read(ROOT/'docs/evidence/load-staged-ssd-profile.json')['cost']
    if args.split == 'independent':
        windows = independent_windows(fixture, model)
        variants = tuple(product((1,12), (F(1),F(5,4))))
    else:
        windows = ((a,b,c,F(1,4),d) for a,b,c,d in inputs(args.split, fixture, model, False))
        variants = tuple(product((1,16), (F(3,4),F(1),F(5,4))))
    for source, window, sharing, spacing, trace in windows:
        for rate, bias in variants:
            cost = dict(fixture['cost'], recompute=grid)
            for kind, config in paths.items():
                scale = rate if kind == 'ssd_to_device' else 1
                cost[kind] = dict(startup_ms=str(F(config['startup_ms'])*scale),
                                  bytes_per_ms=str(F(config['bytes_per_ms'])/scale))
            labels = dict(split=args.split, source=source, window=window, spacing=str(spacing), sharing=sharing, ssd_scale=rate,
                          estimate=str(bias), budget=128, suffix='')
            yield from measure_case(dict(trace,cost=cost), model, 128, {'ssd_to_device':bias}, labels, args.timed, audit=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=('calibration','holdout','independent','witness'), required=True)
    p.add_argument('--timed', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    model = read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    lines = witness(model,args) if args.split == 'witness' else public(model,args)
    count, audits = 0, 0
    with args.out.open('x', newline='', encoding='utf-8') as source:
        writer = None
        for row in lines:
            if writer is None:
                writer = csv.DictWriter(source, fieldnames=row)
                writer.writeheader()
            writer.writerow(row)
            count += 1
            audits += row['audited']
    print(f'{args.split}: {count} result rows; {audits} compact forecast score offsets audited')


if __name__ == '__main__':
    main()
