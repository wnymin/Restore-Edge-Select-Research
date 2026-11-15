"""Compare fixed-continuation queue objectives on paired Bailian windows."""

import argparse
from bisect import bisect_right
from contextlib import ExitStack
import csv
from fractions import Fraction as F
from itertools import islice, product
import json
from pathlib import Path
from statistics import median

from study_bailian import Experiment, ROOT, build, p95, read
from study_boundary_factors import predict


def explain(exp, base, candidate, options, label):
    boundaries = base["boundaries_tokens"]
    alternative = candidate["boundaries_tokens"]
    i = next((j for j, (a, b) in enumerate(zip(boundaries, alternative, strict=True)) if a != b), None)
    if i is None:
        assert base["completion_ms"] == candidate["completion_ms"]
        return dict(comparison=label, first_index=None)
    now = None

    def record(j, time, depth, costs):
        nonlocal now
        if j == i:
            now = time

    exp.run("remaining-wait-live", decision_overrides=dict(enumerate(boundaries[:i])),
            observe_candidates=record)
    visible = bisect_right(exp.arrivals, now)
    fixed = exp.run("fixed", boundaries=[*boundaries[:i], alternative[i], *boundaries[i + 1:]])
    undo = exp.run("snapshot-screened", **options, decision_overrides={i: boundaries[i]})
    assert undo["boundaries_tokens"][:i + 1] == boundaries[:i + 1]

    def deltas(change, reference):
        return [a - b for a, b in zip(change["completion_ms"], reference["completion_ms"], strict=True)]

    delta = deltas(fixed, base)
    row = dict(comparison=label, first_index=i, base_boundary=boundaries[i], queue_boundary=alternative[i],
               visible_later=visible - i - 1, before_ms=float(sum(delta[:i])), current_ms=float(delta[i]),
               seen_later_ms=float(sum(delta[i + 1:visible])), unseen_later_ms=float(sum(delta[visible:])),
               fixed_delta_mean_ms=float(sum(delta) / len(delta)),
               policy_delta_mean_ms=float(candidate["response_mean_ms"] - base["response_mean_ms"]),
               reversing_first_choice_delta_ms=float(undo["response_mean_ms"] - candidate["response_mean_ms"]))
    if label == "serial" and row["policy_delta_mean_ms"] > 0:
        # One-boundary reversals from the final queue plan measure interactions
        # conditional on that plan; they are neither additive nor online.
        best = (F(0), None, None)
        for j, (a, b) in enumerate(zip(boundaries, alternative, strict=True)):
            if a != b:
                mixed = exp.run("fixed", boundaries=[*alternative[:j], a, *alternative[j + 1:]])
                improvement = candidate["response_mean_ms"] - mixed["response_mean_ms"]
                if improvement > best[0]:
                    best = (improvement, j, mixed)
        improvement, j, mixed = best
        if mixed is not None:
            now = None
            i = j
            exp.run("remaining-wait-live", decision_overrides=dict(enumerate(alternative[:i])),
                    observe_candidates=record)
            visible = bisect_right(exp.arrivals, now)
            delta = deltas(mixed, candidate)
            # Remove arrivals the policy could not see. Both candidate traces
            # have the same observed prefix at the decision in either world.
            short = exp._run("fixed", 0, alternative[:visible], 10000, visible)
            short_revert = exp._run("fixed", 0, mixed["boundaries_tokens"][:visible], 10000, visible)
            boundary_outcomes = {}
            for c in exp.eligible[i]:
                plan = [*alternative[:i], c, *alternative[i + 1:]]
                future = exp.run("fixed", boundaries=plan)
                prefix = exp._run("fixed", 0, plan[:visible], 10000, visible)
                boundary_outcomes[c] = dict(full_gain_ms=float(candidate["response_mean_ms"] -
                                                               future["response_mean_ms"]),
                                            truncated_gain_ms=float(short["response_mean_ms"] -
                                                                    prefix["response_mean_ms"]))
            row.update(best_revert_gain_ms=float(improvement), best_revert_index=i,
                       best_revert_visible_later=visible - i - 1, best_revert_current_ms=float(delta[i]),
                       best_revert_seen_and_earlier_ms=float(sum(delta[:i]) + sum(delta[i + 1:visible])),
                       best_revert_unseen_later_ms=float(sum(delta[visible:])),
                       best_revert_truncated_gain_ms=float(short["response_mean_ms"] -
                                                           short_revert["response_mean_ms"]),
                       best_revert_all_boundaries=boundary_outcomes)
    return row


def run_case(exp, key, writer, repeats, diagnostics, lane_probes, lane_core):
    options = dict(score_candidate=lambda r: predict("fifo_snapshot", r), screen_mode="unsettled")
    policies = {
        "single": {},
        "single30": {"min_gain_ms": 30},
        "next1": {"queue_horizon": 1},
        "next4": {"queue_horizon": 4},
        "all": {"queue_horizon": 0},
    }
    if lane_probes:
        policies["next4_robust"] = {"queue_horizon": 4, "future_cost_cases": {
            name: (exp._suffix_times, {}, factors) for name, factors in lane_probes.items()}}
    if lane_core:
        policies = {k: v for k, v in policies.items() if k in {"single", "next4", "next4_robust"}}

    def tail(r):
        return p95([t - a for t, a in zip(r["completion_ms"], exp.arrivals, strict=True)])

    base = exp.run("local-cost")
    timed_base = [exp.run("local-cost", inject_measured=True) for _ in range(repeats)]
    single = None
    for name, extra in policies.items():
        opts = options | extra
        run = exp.run("snapshot-screened", **opts)
        assert exp.run("fixed", boundaries=run["boundaries_tokens"])["completion_ms"] == run["completion_ms"]
        if single is None:
            single = run
        timed = [exp.run("snapshot-screened", **opts, inject_measured=True) for _ in range(repeats)]
        ms = lambda r: sum(v for _, v in r["decision_profile_ns"]) / 1e6
        record = dict(
            **key, policy=name, requests=len(exp.requests), replays=run["decision_replays"],
            forecast_events=run["decision_events"],
            base_mean_ms=float(base["response_mean_ms"]),
            lane_overlap_ms=float(run.get("lane_overlap_ms", 0)),
            both_prefill_ms=float(run.get("overlap_ms_by_path", {}).get("both_lanes_prefill", 0)),
            choices_vs_single=sum(x != y for x, y in
                                  zip(single["boundaries_tokens"], run["boundaries_tokens"], strict=True)),
            mean_gain_ms=float(base["response_mean_ms"] - run["response_mean_ms"]),
            p95_gain_ms=float(tail(base) - tail(run)),
            timed_mean_gain_ms=float(median(b["response_mean_ms"] - r["response_mean_ms"]
                                           for b, r in zip(timed_base, timed, strict=True))) if timed else "",
            timed_p95_gain_ms=float(median(tail(b) - tail(r) for b, r in zip(timed_base, timed, strict=True)))
            if timed else "",
            decision_ms=median(map(ms, timed)) if timed else "",
        )
        writer.writerow(record)
        if diagnostics and name == "all":
            for ref, label in ((base, "serial"), (single, "single")):
                diagnostics.write(json.dumps(key | explain(exp, ref, run, opts, label), default=str) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip", type=int, nargs="+", required=True)
    p.add_argument("--windows", type=int, default=6)
    p.add_argument("--rates", type=F, nargs="+", default=[F(32)])
    p.add_argument("--repeats", type=int, default=1, help="zero omits timing")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--long", type=int, choices=(4096, 8192), default=4096)
    p.add_argument("--prefetch-ahead", type=int, help="override the FIFO prefetch window")
    p.add_argument("--trace", nargs="+", help="run only the named manifest files")
    p.add_argument("--lanes", choices=("shared-compute", "shared", "independent", "coupled"),
                   default="shared-compute")
    p.add_argument("--lane-probes", action="store_true", help="test the two throughput-based envelope endpoints")
    p.add_argument("--lane-core", action="store_true", help="restrict comparisons to single and four-request views")
    p.add_argument("--joint-prefill", action="store_true", help="add measured prefill contention to coupled lanes")
    p.add_argument("--loss-source", type=Path,
                   help="keep cases with >1 ms timed loss for both robust and 30 ms policies in a prior run")
    p.add_argument("--diagnostics", type=Path, help="write first-choice interventions for the selected loss cases")
    p.add_argument("--grid", type=Path, default=ROOT / "docs/evidence/prefill-vulkan-8k.json",
                   help="Continuation cost grid; default reproduces the archived experiment")
    p.add_argument("--io-profile", type=Path, default=ROOT / "docs/evidence/load-staged-ssd-profile.json",
                   help="Transfer cost profile; default reproduces the archived experiment")
    args = p.parse_args()
    if min(args.skip) < 0 or min(args.rates) <= 0 or args.windows < 1 or args.repeats < 0:
        p.error("Expected positive window count and costs and nonnegative offsets and repeats")
    if args.diagnostics and (not args.loss_source or args.lane_core):
        p.error("Diagnostics require --loss-source and the full policy set")
    if args.prefetch_ahead is not None and args.prefetch_ahead < 0:
        p.error("Prefetch window must be nonnegative")
    if (args.lane_probes or args.joint_prefill) and args.lanes != "coupled":
        p.error("Lane probes and joint prefill require the calibrated coupling model")
    def identity(r):
        return tuple(str(r[k]) for k in ("trace", "placement")) + tuple(
            F(str(r[k])) for k in ("window", "spacing", "budget_mib", "io_scale"))
    selected = set()
    if args.loss_source:
        with args.loss_source.open(encoding="utf-8-sig") as source:
            selected = {identity(r) for r in csv.DictReader(source)
                        if max(float(r[k]) for k in ("robust_timed_mean_gain_ms", "unsettled_30_timed_mean_gain_ms")) < -1}

    fixture = read(ROOT / "fixtures/phase1-hand-trace.json")
    if args.prefetch_ahead is not None:
        fixture["scheduling"]["future_window_requests"] = args.prefetch_ahead
    model = read(ROOT / "configs/qwen3.5-4b.json")["models"][0]
    grid = read(args.grid)
    io = read(args.io_profile)["cost"]
    contention = {}
    probes = {}
    if args.lanes == "shared-compute":
        contention["overlap_slowdown"] = read(ROOT / "configs/overlap-observed-upper.json")
    elif args.lanes in {"independent", "coupled"}:
        contention["transfer_topology"] = "per-tier"
    if args.lanes == "coupled":
        with (ROOT / "docs/evidence/boundary-physical-channels.csv").open(encoding="ascii") as source:
            records = list(csv.DictReader(source))
        factors = {p: [1 / F(r["rate_ratio"]) for r in records if r["path"] == p]
                   for p in ("ram", "ssd")}
        profile = lambda fn: {f"{p}_to_device": str(round(fn(v), 3)) for p, v in factors.items()}
        contention["lane_slowdown"] = profile(median)
        if args.lane_probes:
            probes = {"low": profile(min), "high": profile(max)}
        if args.joint_prefill:
            with (ROOT / "docs/evidence/boundary-joint-contention.csv").open(encoding="ascii") as source:
                joint = list(csv.DictReader(source))
            def observed(column, inverse=False):
                factor = median(F(row[column]) for row in joint)
                return str(round(max(F(1), 1 / factor if inverse else factor), 3))
            contention["overlap_slowdown"] = {
                f"{p}_to_device": {"compute": observed(f"{p}_compute_ratio"),
                                   "transfer": observed(f"{p}_single_rate_ratio", inverse=True)}
                for p in ("ram", "ssd")}
            contention["joint_prefill_slowdown"] = {
                "compute": observed("joint_compute_ratio"),
                **{f"{p}_to_device": observed(f"{p}_joint_rate_ratio", inverse=True) for p in ("ram", "ssd")}}
    manifest = read(ROOT / "data/bailian-manifest.json")["files"]
    if args.trace:
        if set(args.trace) - {e["name"] for e in manifest}:
            p.error("Unknown manifest file")
        manifest = [e for e in manifest if e["name"] in args.trace]
    fields = ("cohort", "trace", "window", "placement", "budget_mib", "spacing", "io_scale", "policy",
              "requests", "replays", "forecast_events", "base_mean_ms", "lane_overlap_ms", "both_prefill_ms",
              "choices_vs_single", "mean_gain_ms", "p95_gain_ms",
              "timed_mean_gain_ms", "timed_p95_gain_ms", "decision_ms")
    found = set()
    with ExitStack() as stack:
        out = stack.enter_context(args.out.open("w", newline="", encoding="utf-8"))
        diagnostics = stack.enter_context(args.diagnostics.open("w", encoding="utf-8")) if args.diagnostics else None
        writer = csv.DictWriter(out, fields, lineterminator="\n")
        writer.writeheader()
        for cohort, entry in product(args.skip, manifest):
            trace = entry["name"]
            with (ROOT / "data/raw/bailian" / trace).open(encoding="utf-8") as stream:
                for _ in islice(stream, 352 * cohort):
                    pass
                for window in range(cohort, cohort + args.windows):
                    rows = [json.loads(line, parse_float=str) for line in islice(stream, 352)]
                    if len(rows) != 352:
                        raise ValueError(f"Incomplete window: {trace}/{window}")
                    for placement, spacing in product(("split", "ram", "ssd"), (F(1), F(1, 4))):
                        data = build(rows, 256, fixture, model, spacing, placement, args.long)
                        for rate, budget in product(args.rates, (128, 384)):
                            key = dict(cohort=cohort, trace=trace, window=window, placement=placement,
                                       budget_mib=budget, spacing=float(spacing), io_scale=rate)
                            ident = identity(key)
                            if args.loss_source and ident not in selected:
                                continue
                            found.add(ident)
                            cost = dict(fixture["cost"], recompute=grid)
                            for path, spec in io.items():
                                factor = F(1) if path == "device_copy" else rate
                                cost[path] = dict(startup_ms=str(F(spec["startup_ms"]) * factor),
                                                  bytes_per_ms=str(F(spec["bytes_per_ms"]) / factor))
                            exp = Experiment(dict(data, cost=cost, **contention), model, budget)
                            run_case(exp, key, writer, args.repeats, diagnostics, probes, args.lane_core)
                    out.flush()
    if args.loss_source and not found:
        p.error("No cases matched the prior losses")


if __name__ == "__main__":
    main()
