"""Audit when a cheap Phase 1 certificate can stop before queue lookahead."""

import argparse
from bisect import bisect_right
from collections import Counter
import csv
from fractions import Fraction as F
from itertools import islice, product
import json
from pathlib import Path
from time import perf_counter_ns

from scripts.cpu.replay.study_bailian import build
from scripts.cpu.replay.study_boundary_factors import predict
from scripts.cpu.lookahead.screening.study_lookahead_fifo_score import ActiveTailScore, opts
from scripts.cpu.lookahead.screening.study_lookahead_phase1 import Experiment, ROOT, read, trace_for, without_cross_request_sharing


class Trigger(ActiveTailScore):
    """Private adapter for the fixed, single-lane Phase 1 scoring call site."""

    def __init__(self, trace, model, capacity, *, audit=False):
        started = perf_counter_ns()
        super().__init__(trace, model, capacity, audit=audit)
        self.eligible_heads = [i for i, options in enumerate(self.eligible) if len(options) > 1]
        self.last_early = []
        last = -1
        for i, options in enumerate(self.eligible):
            def isolated(c):
                deps = self.candidates[i][c]
                copy = (self.path_time('device_copy', self.objects[deps[-1]]['bytes'])
                        if deps and c < self.lengths[i] and deps[-1] in self.initial else 0)
                return self._suffix_times[i][c] + copy + sum(map(self.transfer_time, self.missing[i][c])), -c
            choice = min(options, key=isolated)
            if self.can_prefetch[i][choice]:
                last = i
            self.last_early.append(last)
        self.any_early = [any(self.can_prefetch[i][c] for c in options)
                          for i, options in enumerate(self.eligible)]
        self.precompute_ns = perf_counter_ns() - started

    def run_trigger(self, estimate, name, *, measure=False, profile=False):
        self.stats.clear()
        self.mode = name
        expected = iter(self.eligible_heads)
        self.cut = None
        self.distance = None
        self.proposal = None
        self.open_horizon = None
        self.minima = {}

        def select(rows, fallback, later):
            i = self.i = next(expected)
            assert rows.keys() == set(self.eligible[i])
            self.row = rows[fallback]
            self.fallback = fallback
            self.distance = self.row['head_distance']
            self.cut = self.offset = None
            self.n_visible = i + 1 + later
            self.stats['multi'] += 1
            self.minima = {}
            self.proposal = None
            if name in {'legacy-admit', 'certified'}:
                self.proposal = self.certificate(rows, fallback, i, self.n_visible)
            if self.proposal:
                reason, best = self.proposal
                self.stats[reason] += 1
                if best == fallback:
                    self.stats['skip_all'] += 1
                    self.open_horizon = 0 if self.audit else None
                else:
                    # Closed-form order says the local challenger wins. Keep
                    # the ordinary target-only evaluation to select the action.
                    self.stats['skip_peers'] += 1
                    self.open_horizon = 0 if self.audit else 'current'
            else:
                self.open_horizon = 'current' if name == 'target' else 0
            return self.open_horizon

        def order(row):
            if name == 'legacy' or name == 'legacy-admit' and not self.proposal:
                return predict('fifo_snapshot', row)
            return int(row is self.row)  # Ensure every candidate is scored by an actual rollout.

        start = perf_counter_ns()
        result = self.run('snapshot-screened', **(opts(0, estimate) | {'screen_selector': select, 'score_candidate': order}),
                          profile_decisions=profile, inject_measured=measure)
        assert next(expected, None) is None
        return result, (perf_counter_ns()-start) / 1e6

    def certificate(self, rows, fallback, i, n):
        if self.last_early[n-1] > i:
            return None
        r = rows[fallback]
        if not r['head_distance']:
            if r['inflight_ms'] or r['queue_debt_ms']:
                return None
            return ('head', fallback)
        if not self.any_early[i]:
            return ('no-admission', fallback)
        slack = r['head_slack_ms']
        if r['head_distance'] != 1 or not slack or r['inflight_ms'] or r['queue_debt_ms']:
            return None
        if i - (i-1) > self.window:
            return None

        def exact(c):
            row = rows[c]
            cost = row['serial_compute_ms'] + row['serial_copy_ms']
            io = row['incremental_ram_ms'] + row['incremental_ssd_ms']
            return cost + (max(0, io - slack) if row['prefetch_fits'] else row['serial_io_ms'])
        best = min(rows, key=lambda c: (exact(c), -c))
        return ('one-running', best if exact(best) < exact(fallback) else fallback)

    def _run(self, policy, threshold_tokens, boundaries, max_combinations, n, **kwargs):
        state = kwargs.get('snapshot')
        if state is None:
            return Experiment._run(self, policy, threshold_tokens, boundaries, max_combinations, n, **kwargs)
        assert state[1] + self.distance == self.i
        if self.audit and self.proposal:
            assert n == self.n_visible
            assert not any(j > self.i for j in state[3])
        short = self.open_horizon == 'current'
        if self.mode in {'compact', 'certified', 'legacy-admit'} and not short:
            out = super()._run(policy, threshold_tokens, boundaries, max_combinations, n, **kwargs)
        else:
            out = Experiment._run(self, policy, threshold_tokens, boundaries, max_combinations, n, **kwargs)
        if self.audit and self.proposal:
            c = boundaries[self.i]
            self.minima[c] = sum(out['completion_ms'])
            if len(self.minima) == len(self.eligible[self.i]):
                winner = min(self.minima, key=lambda c: (self.minima[c], -c))
                if self.minima[winner] >= self.minima[self.fallback]:
                    winner = self.fallback
                assert self.proposal[1] == winner, (self.proposal, winner, self.i, self.minima)
                self.stats['certificates_audited'] += 1
        return out


def cases(args, model):
    if args.split == 'witness':
        for mode, capacity, suffix, bias in product(('distinct','alias'), (0, 128, 248, 256),
                                                    (0, 16), (F(3,4), F(1), F(5,4))):
            trace = trace_for(mode, model, free_case=True)
            trace['cost']['ssd_to_device']['bytes_per_ms'] = str(F(trace['cost']['ssd_to_device']['bytes_per_ms'])*2)
            for r in trace['requests'][1:]:
                r['arrival_ms'] = 1
            for j in range(suffix):
                trace['requests'].append(dict(id=f'T{j}', arrival_ms=1,
                                              input_token_runs=[dict(token_id=10000+j, count=32)],
                                              candidates=[dict(boundary_tokens=0, load_order=[])]))
            yield trace, dict(split=args.split, source=mode, window='', spacing='same', sharing=mode,
                              ssd_scale='1/2', estimate=str(bias), budget=capacity, suffix=suffix)
        return
    fixture = read(ROOT/'fixtures/phase1-hand-trace.json')
    grid = read(ROOT/'docs/evidence/prefill-vulkan-4k.json')
    paths = read(ROOT/'docs/evidence/load-staged-ssd-profile.json')['cost']
    sources = ('qwen_coder_blksz_16.jsonl', 'qwen_traceA_blksz_16.jsonl',
               'qwen_traceB_blksz_16.jsonl', 'qwen_thinking_blksz_16.jsonl')
    for i, source in enumerate(sources):
        # Consecutive unused windows fixed by arithmetic, not by policy outcomes.
        window = (4 if args.split == 'calibration' else 22) + i
        with (ROOT/'data/raw/bailian'/source).open(encoding='utf-8') as f:
            rows = [json.loads(line, parse_float=str) for line in islice(f, 352*window, 352*(window+1))]
        assert len(rows) == 352
        for spacing in (F(1), F(1, 32)):
            trace = build(rows, 256, fixture, model, spacing, 'split', 3584)
            for sharing, rate, bias, capacity in product(('native','disabled'), (1,12), (F(1),F(5,4)), (128,256)):
                variant = trace if sharing == 'native' else without_cross_request_sharing(trace)
                cost = dict(fixture['cost'], recompute=grid)
                for kind, config in paths.items():
                    scale = rate if kind == 'ssd_to_device' else 1
                    cost[kind] = dict(startup_ms=str(F(config['startup_ms'])*scale),
                                      bytes_per_ms=str(F(config['bytes_per_ms'])/scale))
                yield dict(variant, cost=cost), dict(split=args.split, source=source, window=window,
                                                    spacing=str(spacing), sharing=sharing, ssd_scale=rate,
                                                    estimate=str(bias), budget=capacity, suffix='')


def study(model, args):
    names = ('target','full','compact','certified','legacy','legacy-admit')
    for trace, labels in cases(args, model):
        capacity = labels['budget']
        estimate = {'ssd_to_device': F(labels['estimate'])}
        audit = Trigger(trace,model,capacity,audit=True)
        audited, _ = audit.run_trigger(estimate,'certified')
        results = {}
        measured = {}
        for name in names:
            exp = Trigger(trace,model,capacity)
            out, wall = exp.run_trigger(estimate,name,profile=args.timed)
            results[name] = out, wall, exp.stats.copy(), exp.precompute_ns
        if args.timed:
            timing_order = names[::-1] if args.reverse_timing else names
            for name in timing_order:
                exp = Trigger(trace,model,capacity)
                actual, _ = exp.run_trigger(estimate,name,measure=True)
                measured[name] = actual
        target = results['target'][0]
        full = results['full'][0]
        for same in ('compact','certified'):
            assert results[same][0]['boundaries_tokens'] == full['boundaries_tokens'] == audited['boundaries_tokens']
            assert results[same][0]['completion_ms'] == full['completion_ms'] == audited['completion_ms']
        oracle = Experiment(trace,model,capacity).offline() if args.split == 'witness' else None
        for name, (out, wall, stats, startup) in results.items():
            delta = [a-b for a,b in zip(out['completion_ms'],target['completion_ms'],strict=True)]
            actual = measured.get(name)
            yield labels | dict(policy=name, requests=len(delta), multi=stats['multi'],
                                skip_all=stats['skip_all'], skip_peers=stats['skip_peers'],
                                head=stats['head'], no_admission=stats['no-admission'], running=stats['one-running'],
                                audit=audit.stats['certificates_audited'] if name=='certified' else 0,
                                replays=out['decision_replays'], events=out['decision_events'],
                                changed_vs_target=sum(a!=b for a,b in zip(out['boundaries_tokens'],target['boundaries_tokens'],strict=True)),
                                changed_vs_full=sum(a!=b for a,b in zip(out['boundaries_tokens'],full['boundaries_tokens'],strict=True)),
                                response_gain_ms=float(-sum(delta)), worst_vs_target_ms=float(max(0,*delta)),
                                response_sum_ms=float(out['response_sum_ms']),
                                joint_regret_ms=float(out['response_sum_ms']-oracle['response_sum_ms']) if oracle else '',
                                prep_ms=startup/1e6, wall_ms=wall,
                                decision_ms=sum(v for _,v in out['decision_profile_ns'])/1e6 if args.timed else '',
                                decision_injected_ms=sum(v for _,v in actual['decision_profile_ns'])/1e6 if actual else '',
                                response_injected_ms=float(actual['response_sum_ms']) if actual else '')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=('witness','calibration','holdout'), required=True)
    p.add_argument('--timed', action='store_true')
    p.add_argument('--reverse-timing', action='store_true')
    p.add_argument('--out', required=True, type=Path)
    args=p.parse_args()
    model=read(ROOT/'configs/qwen3.5-4b.json')['models'][0]
    with args.out.open('x',newline='',encoding='utf-8') as fd:
        writer=None
        count=0
        for row in study(model,args):
            if writer is None:
                writer=csv.DictWriter(fd,fieldnames=row)
                writer.writeheader()
            writer.writerow(row)
            count+=1
    print(f'{args.split}: {count} result rows')


if __name__=='__main__':
    main()
