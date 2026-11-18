"""Check exact serial-tail projection in Phase 1 forecasts and public replay."""

import argparse
from collections import Counter
from copy import deepcopy
import csv
from fractions import Fraction as F
from itertools import islice, product
import json
from pathlib import Path
from time import perf_counter_ns

from scripts.cpu.replay.study_bailian import Experiment, ROOT, build, p95, read
from scripts.cpu.replay.study_boundary_factors import predict
from scripts.cpu.replay.study_boundary_joint_phase1 import trace_for


class SerialTailExperiment(Experiment):
    """Research adapter: keep the production-independent simulator unchanged."""

    def __init__(self, trace, model, capacity, *, audit=False):
        super().__init__(trace, model, capacity)
        if self.per_tier or self.dynamic or self.host_cache_bytes or self.release_at != 'start':
            raise ValueError('Serial suffix proof requires stationary single-lane Phase 1 semantics')
        self.audit = audit
        self.stats = Counter()
        self.missing = [{b: tuple(k for k in deps if k not in self.initial) for b, deps in choices.items()}
                        for choices in self.candidates]
        self.can_prefetch = [{b: bool(deps) and self.size(deps) <= self.budget for b, deps in choices.items()}
                             for choices in self.missing]
        self.work_cache = {}

    def _run(self, policy, threshold_tokens, boundaries, max_combinations, n, **kwargs):
        engine = super()._run
        state = kwargs.get('snapshot')
        if (state is None or policy != 'fixed' or not kwargs.get('lean_rollout')
                or kwargs.get('completion_cutoff') is not None or kwargs.get('future_cost') is not None
                or kwargs.get('prefetch_budget') not in (None, self.budget)):
            return engine(policy, threshold_tokens, boundaries, max_combinations, n, **kwargs)
        now, head, visible, admitted = state[:4]
        assert visible == n
        self.stats['forecasts'] += 1
        # Preserve the active head and all possible prefetching requests. No
        # discarded request may already own a reservation in this snapshot.
        cut = n
        while cut > head + 1 and cut - 1 not in admitted and not self.can_prefetch[cut - 1][boundaries[cut - 1]]:
            cut -= 1
        if cut == n:
            return engine(policy, threshold_tokens, boundaries, max_combinations, n, **kwargs)
        self.stats['projected_forecasts'] += 1
        self.stats['projected_requests'] += n - cut
        prefix = engine(policy, threshold_tokens, boundaries[:cut], max_combinations, cut,
                        **(kwargs | {'snapshot': (*state[:2], cut, *state[3:])}))
        result = prefix['completion_ms']
        estimate = kwargs.get('estimate') or {}
        speeds = kwargs['forecast_speed']
        key = (tuple(sorted(estimate.items())), tuple(sorted(speeds.items())))
        cache = self.work_cache.setdefault(key, {})
        end = result[-1]
        for i in range(cut, n):
            b = boundaries[i]
            if (i, b) not in cache:
                miss = self.missing[i][b]
                deps = self.candidates[i][b]
                cost = self._suffix_times[i][b] * estimate.get('compute', 1)
                if deps and deps[-1] in self.initial and b < self.lengths[i]:
                    cost += self.path_time('device_copy', self.objects[deps[-1]]['bytes']) * estimate.get('device_copy', 1)
                for obj in miss:
                    path = {'RAM': 'ram_to_device', 'SSD': 'ssd_to_device'}[self.objects[obj]['tier']]
                    cost += self._transfer_times[obj] * estimate.get(path, 1) / speeds[path]
                cache[i, b] = cost
            end = max(end, self.arrivals[i]) + cache[i, b]
            result.append(end)
        if self.audit:
            full = engine(policy, threshold_tokens, boundaries, max_combinations, n, **kwargs)
            assert result == full['completion_ms'], (head, cut, n, boundaries)
        return prefix


def opts(horizon, estimate):
    return dict(score_candidate=lambda row: predict('fifo_snapshot', row), screen_mode='classified',
                screen_selector=lambda *_: horizon, screen_search='lean', estimate=estimate)


def run_policy(exp, policy, estimate, *, timed=False):
    horizon = {'target': 'current', 'fixed4': 4, 'all': 0, 'tail': 0}[policy]
    before = perf_counter_ns()
    out = exp.run('snapshot-screened', **opts(horizon, estimate), inject_measured=timed)
    return out, perf_counter_ns() - before


def small_audit(model):
    count = 0
    for mode, capacity, bias in product(('distinct', 'alias'), (0, 80, 248, 256), (F(3, 4), F(1), F(5, 4))):
        trace = trace_for(mode, model, free_case=True)
        for r in trace['requests'][1:]:
            r['arrival_ms'] = 1   # All three challenger requests are already visible.
        estimate = {'ssd_to_device': bias}
        fixed = Experiment(trace, model, capacity)
        tail = SerialTailExperiment(trace, model, capacity, audit=True)
        actual = fixed.offline()
        run, _ = run_policy(tail, 'tail', estimate)
        oracle, _ = run_policy(fixed, 'all', estimate)
        assert run['boundaries_tokens'] == oracle['boundaries_tokens']
        assert run['completion_ms'] == oracle['completion_ms']
        assert run['response_sum_ms'] >= actual['response_sum_ms']
        count += tail.stats['projected_forecasts']
    assert count
    print(f'24 small joint-oracle checks; {count} individual forecasts projected and verified', flush=True)


def without_cross_request_sharing(trace):
    trace = deepcopy(trace)
    objects = {x['id']: x for x in trace['objects']}
    unique = {}
    for r in trace['requests']:
        for c in r['candidates']:
            names = []
            for key in c['load_order']:
                new = f"{r['id']}:{key}"
                if new not in unique:
                    unique[new] = objects[key] | {'id': new}
                names.append(new)
            c['load_order'] = names
    trace['objects'] = list(unique.values())
    return trace


def inputs(split, fixture, model, pilot):
    # Separate trace sources and raw windows, not windows chosen by the outcome.
    groups = {'calibration': [('qwen_coder_blksz_16.jsonl', 34), ('qwen_traceA_blksz_16.jsonl', 38)],
              'holdout': [('qwen_traceB_blksz_16.jsonl', 44), ('qwen_thinking_blksz_16.jsonl', 23)]}
    history, width = 256, 352
    for source, window in groups[split][:1] if pilot else groups[split]:
        with (ROOT / 'data/raw/bailian' / source).open(encoding='utf-8') as f:
            rows = [json.loads(line, parse_float=str)
                    for line in islice(f, width * window, width * (window + 1))]
        if len(rows) != width:
            raise ValueError(f'Incomplete trace window: {source} {window}')
        native = build(rows, history, fixture, model, F(1, 4), 'split', 3584)
        for sharing in ('native', 'disabled'):
            yield source, window, sharing, native if sharing == 'native' else without_cross_request_sharing(native)


def study(args, model):
    fixture = read(ROOT / 'fixtures/phase1-hand-trace.json')
    grid = read(ROOT / 'docs/evidence/prefill-vulkan-4k.json')
    paths = read(ROOT / 'docs/evidence/load-staged-ssd-profile.json')['cost']
    for source, window, sharing, trace in inputs(args.split, fixture, model, args.pilot):
        for rate, bias in product((1, 16), (F(3, 4), F(1), F(5, 4)) if not args.pilot else (F(1),)):
            cost = dict(fixture['cost'], recompute=grid)
            for kind, config in paths.items():
                scale = rate if kind == 'ssd_to_device' else 1
                cost[kind] = dict(startup_ms=str(F(config['startup_ms']) * scale),
                                  bytes_per_ms=str(F(config['bytes_per_ms']) / scale))
            case = dict(trace, cost=cost)
            label = dict(split=args.split, source=source, window=window, sharing=sharing, ssd_scale=rate,
                         ssd_estimate=str(bias), budget=args.budget)
            print(*label.values(), flush=True)
            estimate = {'ssd_to_device': bias}
            baseline = None
            for policy in ('target', 'fixed4', 'all', 'tail'):
                exp = (SerialTailExperiment if policy == 'tail' else Experiment)(case, model, args.budget)
                out, elapsed = run_policy(exp, policy, estimate)
                if policy == 'target':
                    baseline = out
                if policy == 'all':
                    full = out
                if policy == 'tail':
                    assert full['boundaries_tokens'] == out['boundaries_tokens']
                    assert full['completion_ms'] == out['completion_ms']
                stats = exp.stats.copy() if policy == 'tail' else Counter()
                timed, _ = run_policy(exp, policy, estimate, timed=True) if args.timed else (None, None)
                delta = [a-b for a,b in zip(out['completion_ms'], baseline['completion_ms'], strict=True)]
                response = [a-b for a,b in zip(out['completion_ms'], exp.arrivals, strict=True)]
                yield dict(**label, policy=policy, requests=len(delta), changes=sum(d != 0 for d in delta),
                           mean_ms=float(out['response_mean_ms']), p95_ms=float(p95(response)),
                           worst_vs_target_ms=float(max(0, *delta)), replays=out['decision_replays'],
                           events=out['decision_events'], projected_forecasts=stats['projected_forecasts'],
                           projected_requests=stats['projected_requests'], wall_ms=elapsed/1e6,
                           timed_mean_ms=float(timed['response_mean_ms']) if timed else '',
                           decision_cpu_ms=sum(t for _, t in timed['decision_profile_ns']) / 1e6 if timed else '')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=('calibration', 'holdout'), default='holdout')
    p.add_argument('--budget', type=int, default=128)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--timed', action='store_true')
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    model = read(ROOT / 'configs/qwen3.5-4b.json')['models'][0]
    small_audit(model)
    if not args.out:
        return
    with args.out.open('x', newline='', encoding='utf-8') as out:
        writer = None
        for row in study(args, model):
            if writer is None:
                writer = csv.DictWriter(out, fieldnames=row.keys())
                writer.writeheader()
            writer.writerow(row)
            out.flush()


if __name__ == '__main__':
    main()
