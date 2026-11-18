"""Compare recovery rankings from a common snapshot under one and two prefill slots."""

import argparse
from collections import defaultdict
import csv
from fractions import Fraction as F
from itertools import islice, product
import json
from math import log
from pathlib import Path
from statistics import mean, median

from scripts.cpu.replay.study_bailian import Experiment, ROOT, build, p95, read
from restore_edge_select.batch_replay import BatchReplay


def calibrate(path):
    samples = defaultdict(list)
    with path.open(encoding="ascii") as source:
        for row in csv.DictReader(source):
            first = float(row["sequential_common_arrival_sum_ms"]) - float(row["sequential_span_ms"])
            solo = [first, float(row["sequential_sum_response_ms"]) - first]
            ends = [solo[i] * float(row[f"slot{i}_response_ratio"]) for i in (0, 1)]
            suffix = [int(row[f"slot{i}_suffix"]) for i in (0, 1)]
            order = sorted(range(2), key=lambda i: (suffix[i], solo[i]))
            j, k = sorted(range(2), key=ends.__getitem__)
            # After the first completion the peer resumes its solo rate.
            rate = [0., 0.]
            rate[j] = solo[j] / ends[j]
            rate[k] = (solo[k] - (ends[k] - ends[j])) / ends[j]
            # Equal shapes need a slot-neutral rate; label noise is not a feature.
            if suffix[0] == suffix[1]:
                rate = [mean(rate)] * 2
            # Fixed-denominator progress rates keep exact event times tractable.
            rates = tuple(F(round(100 * min(1., max(.05, rate[i]))), 100) for i in order)
            samples[tuple(suffix[i] for i in order)].append((rates, tuple(solo[i] for i in order),
                                                             tuple(ends[i] for i in order)))
    return samples


def main_rates(samples, mode):
    levels = {}
    for shape, values in samples.items():
        values = sorted((value[0] for value in values), key=lambda x: (sum(x), x))
        levels[shape] = (tuple(median(x[i] for x in values) for i in (0, 1)) if mode == "median"
                         else values[0 if mode == "slow" else -1])

    def rates(pair):
        order = sorted(range(2), key=lambda i: (pair[i][1], pair[i][0]))
        sizes = tuple(pair[i][1] for i in order)
        nearest = min(levels, key=lambda shape: sum(abs(log(a / b)) for a, b in zip(shape, sizes)))
        if sizes[0] == sizes[1]:
            return (mean(levels[nearest]),) * 2
        mapped = dict(zip(order, levels[nearest], strict=True))
        return mapped[0], mapped[1]

    return rates


def gain(base, run, n, arrivals):
    old = [base.done[i] - arrivals[i] for i in range(n)]
    new = [run.done[i] - arrivals[i] for i in range(n)]
    return dict(mean_gain_ms=float(mean(old) - mean(new)), p95_gain_ms=float(p95(old) - p95(new)))


def run(exp, samples, key, record, controls):
    n = len(exp.requests)
    candidates = exp.eligible
    models = {mode: BatchReplay(exp, main_rates(samples, mode)) for mode in ("median", "slow", "fast")}
    nominal = models["median"]
    base = nominal.run()
    original = exp.run("local-cost")
    assert [base.selected[i] for i in range(n)] == original["boundaries_tokens"]
    assert [base.done[i] for i in range(n)] == original["completion_ms"]
    assert [base.reserved.get(i) for i in range(n)] == original["prefetch_reserved_at_ms"]

    checks = defaultdict(int)

    def objective(state, i, c, later, slots, forecast):
        r = forecast.rollout(state, i, c, later, slots)
        return sum(r.done[j] for j in range(later) if j not in state.done)

    def observe(s, i):
        c = nominal.baseline[i]
        if len(candidates[i]) == 1:
            return c
        later = min(s.arrived, i + 5)
        checks["snapshots"] += 1
        scores = {slots: {b: objective(s, i, b, later, slots, nominal) for b in candidates[i]}
                  for slots in (1, 2)}
        picks = [min(x, key=lambda b: (x[b], -b)) for x in scores.values()]
        if picks[0] != picks[1]:
            a, b = picks
            one, two = (scores[q][alt] - scores[q][pref] for q, alt, pref in ((1, b, a), (2, a, b)))
            if min(one, two) > 0:
                checks["reversal"] += 1
                checks["reversal_gt1"] += min(one, two) > 1
                checks["reversal_gt30"] += min(one, two) > 30
                if min(one, two) > 1 and not any(x["io_scale"] == key["io_scale"]
                                                and x["kind"] == "rank" for x in controls):
                    detail = {}
                    for slots, option in product((1, 2), (a, b)):
                        ending = nominal.rollout(s, i, option, later, slots)
                        detail[f"{slots}:{option}"] = {
                            "reserve_after_ms": float(ending.reserved[i] - s.now) if i in ending.reserved else None,
                            "prefill_after_ms": float(ending.started[i] - s.now),
                            "finish_after_ms": float(ending.done[i] - s.now),
                            "earlier_finish_after_ms": {j: float(ending.done[j] - s.now)
                                                        for j in range(i) if j not in s.done},
                        }
                    controls.append(key | dict(kind="rank", request=i, visible=later - i, base=c, one=a, two=b,
                                               one_gap_ms=float(one), two_gap_ms=float(two), time_ms=float(s.now),
                                               branches=detail))
        return c

    assert nominal.run(choose=observe).done == base.done
    policies = [("median", 1, 4, "point"), ("median", 2, 0, "point")]
    policies += [(mode, 2, 4, policy) for mode, policy in product(models, ("point", "guard"))]
    for mode, slots, horizon, policy in policies:
        actual = models[mode]
        forecast = nominal
        static = actual.run(slots=slots)
        changed = defaultdict(int)

        def choose(s, i):
            c = nominal.baseline[i]
            if len(candidates[i]) == 1:
                return c
            later = min(s.arrived, i + 1 + horizon)
            if horizon:
                score = {b: objective(s, i, b, later, slots, forecast) for b in candidates[i]}
            else:
                # For the current-only objective, discard effects on older requests.
                score = {b: forecast.rollout(s, i, b, later, slots).done[i] for b in candidates[i]}
            best = min(score, key=lambda b: (score[b], -b))
            if score[best] >= score[c]:
                return c
            if policy == "guard" and any(objective(s, i, best, later, slots, model) >=
                                         objective(s, i, c, later, slots, model)
                                         for name, model in models.items() if name != "median"):
                changed["rejected"] += 1
                return c
            changed["changed"] += 1
            return best

        chosen = actual.run(slots=slots, choose=choose)
        assert chosen.done == actual.run(slots=slots, fixed=chosen.selected).done
        if not horizon and key["io_scale"] == 1 and (sum(chosen.done.values()) - sum(static.done.values())) > n:
            if not any(x["kind"] == "local_externality" for x in controls):
                i = next(j for j in range(n) if chosen.selected[j] != nominal.baseline[j])
                plan = list(nominal.baseline)
                plan[i] = chosen.selected[i]
                fixed = actual.run(slots=2, fixed=plan)
                delta = [fixed.done[j] - static.done[j] for j in range(n)]
                controls.append(key | dict(kind="local_externality", request=i, base=nominal.baseline[i],
                                           alternative=plan[i], earlier_ms=float(sum(delta[:i])),
                                           current_ms=float(delta[i]), later_ms=float(sum(delta[i + 1:])),
                                           fixed_mean_delta_ms=float(mean(delta)),
                                           policy_mean_delta_ms=float(
                                               (sum(chosen.done.values()) - sum(static.done.values())) / n)))
        record(dict(**key, rate_mode=mode, slots=slots, horizon=horizon, policy=policy,
                    **gain(static, chosen, n, exp.arrivals), **{
                        "base_mean_ms": float(mean(static.done[i] - exp.arrivals[i] for i in range(n))),
                        "batch_ms": float(chosen.batch_ms), "snapshots": checks["snapshots"],
                        "reversals": checks["reversal"], "reversals_gt1": checks["reversal_gt1"],
                        "reversals_gt30": checks["reversal_gt30"], "changed": changed["changed"],
                        "rejected": changed["rejected"],
                    }))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", required=True, type=int, choices=(18, 24))
    p.add_argument("--windows", type=int, default=6)
    p.add_argument("--start-window", type=int, help="override the first window for attribution")
    p.add_argument("--trace")
    p.add_argument("--no-io-contention", action="store_true")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--grid", type=Path, default=ROOT / "docs/evidence/prefill-vulkan-8k.json",
                   help="Continuation cost grid; default reproduces the archived experiment")
    p.add_argument("--io-profile", type=Path, default=ROOT / "docs/evidence/load-staged-ssd-profile.json",
                   help="Transfer cost profile; default reproduces the archived experiment")
    args = p.parse_args()
    start = args.cohort if args.start_window is None else args.start_window
    if start < 0 or args.windows < 1:
        p.error("Expected nonnegative start and positive window count")
    cap = 4096 if args.cohort == 18 else 8192
    fixture = read(ROOT / "fixtures/phase1-hand-trace.json")
    model = read(ROOT / "configs/qwen3.5-4b.json")["models"][0]
    grid = read(args.grid)
    io = read(args.io_profile)["cost"]
    overlap = read(ROOT / "configs/overlap-observed-upper.json")
    samples = calibrate(ROOT / "docs/evidence/boundary-batched-prefill-warm.csv")
    sources = [e for e in read(ROOT / "data/bailian-manifest.json")["files"]
               if not args.trace or e["name"] == args.trace]
    if not sources:
        p.error("Unknown trace name")
    fields = ("cohort", "trace", "window", "spacing", "budget_mib", "io_scale", "rate_mode", "slots", "horizon", "policy",
              "mean_gain_ms", "p95_gain_ms", "base_mean_ms", "batch_ms", "snapshots", "reversals",
              "reversals_gt1", "reversals_gt30", "changed", "rejected")
    controls = []
    with args.out.open("w", encoding="utf-8", newline="") as dest:
        writer = csv.DictWriter(dest, fields, lineterminator="\n")
        writer.writeheader()
        for entry in sources:
            trace = entry["name"]
            with (ROOT / "data/raw/bailian" / trace).open(encoding="utf-8") as source:
                for _ in islice(source, 352 * start):
                    pass
                for window in range(start, start + args.windows):
                    rows = [json.loads(line, parse_float=str) for line in islice(source, 352)]
                    assert len(rows) == 352
                    for spacing in (F(1), F(1, 4)):
                        data = build(rows, 256, fixture, model, spacing, "split", cap)
                        # The second slot gets a separate worst-case private budget.
                        data["memory"]["device_mib"] += data["memory"]["execution_mib"]
                        for rate, budget in product((1, 32), (128, 384)):
                            cost = dict(fixture["cost"], recompute=grid)
                            for path, entry in io.items():
                                f = 1 if path == "device_copy" else rate
                                cost[path] = dict(startup_ms=str(F(entry["startup_ms"]) * f),
                                                  bytes_per_ms=str(F(entry["bytes_per_ms"]) / f))
                            profile = {} if args.no_io_contention else overlap
                            exp = Experiment(dict(data, cost=cost, overlap_slowdown=profile), model, budget)
                            run(exp, samples, dict(cohort=args.cohort, trace=trace, window=window, spacing=float(spacing),
                                          budget_mib=budget, io_scale=rate), writer.writerow, controls)
                    dest.flush()
    args.out.with_suffix(".json").write_text(json.dumps(controls, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
