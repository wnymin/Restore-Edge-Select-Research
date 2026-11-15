"""Paired check of decision-time load classes and policy switches on Bailian."""

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction as F
from itertools import islice, product
import json
from pathlib import Path
from statistics import mean

from study_bailian import Experiment, ROOT, build, p95, read
from study_boundary_factors import predict


CLASSES = ("idle", "light_io", "capacity", "transfer_queue", "reuse", "overlap", "uncoupled")
EXACT_SEARCH = {"lean", "compute", "cutoff", "entry"}


def signals(rows, base, later, budget):
    b = rows[base]
    alternatives = [v for c, v in rows.items() if c]
    serial = b["serial_compute_ms"] + b["serial_copy_ms"] + b["serial_io_ms"]
    load = max((r["serial_io_ms"] for r in alternatives), default=0)
    work = b["committed_ahead_compute_ms"] + b["head_slack_ms"]
    debt = b["queue_debt_ms"] + b["inflight_ms"]
    ahead = b["head_distance"]
    any_reuse = any(r["ready_bytes"] or r["shared_reservation_bytes"] or
                    (r["prefetch_fits"] and r["active_new_kv_bytes"]) for r in alternatives)
    blocked = ahead and any(not r["prefetch_fits"] and r["extra_reserve_bytes"] <= budget
                            for r in alternatives)
    relative = load / max(serial, 1)
    if not (ahead or later or debt or any_reuse):
        label = "idle"
    elif 8 * load < serial:
        label = "light_io"
    elif blocked:
        label = "capacity"
    elif debt:
        label = "transfer_queue"
    elif any_reuse:
        label = "reuse"
    elif work:
        label = "overlap"
    else:
        label = "uncoupled"
    return label, relative, ahead, work, debt


def run_case(exp, key, results, snapshots, zero_only, policies):
    scorer = lambda row: predict("fifo_snapshot", row)
    opts = dict(score_candidate=scorer, screen_mode="unsettled")
    klass = dict(score_candidate=scorer, screen_mode="classified")
    fixture = exp.run("local-cost")
    wall_base = exp.run("local-cost", inject_measured=True) if not zero_only else None
    diag = defaultdict(Counter)

    def observe(rows, base, later, gain):
        label, relative, ahead, work, debt = signals(rows, base, later, exp.budget)
        for scope in (label, "all"):
            c = diag[scope]
            c["decisions"] += 1
            c["gains"] += gain > 0
            c["gains_gt5"] += gain > 5
            c["sum_gain"] += float(gain)
            c["relative"] += float(relative)
            c["ahead"] += ahead
            c["later"] += later
            c["slack_ms"] += float(work)
            c["debt_ms"] += float(debt)
            if (label in {"capacity", "transfer_queue", "reuse", "overlap"}) and gain > 5:
                c["routing_captures_gt5"] += 1
            if ahead and gain > 5:
                c["depth_captures_gt5"] += 1

    if snapshots is not None:
        inspected = exp.run("snapshot-screened", **klass, screen_selector=lambda *_: 4,
                            screen_observer=observe, min_gain_ms=10**12)
        assert inspected["boundaries_tokens"] == fixture["boundaries_tokens"]
        assert inspected["completion_ms"] == fixture["completion_ms"]
    for label, c in diag.items():
        snapshots.writerow(dict(**key, label=label, **{k: c[k] for k in
            ("decisions", "gains", "gains_gt5", "sum_gain", "relative", "ahead", "later",
             "slack_ms", "debt_ms", "routing_captures_gt5", "depth_captures_gt5")}))

    def selector(mode, calls):
        def route(rows, base, later):
            if mode == "depth":
                view = 4 if rows[base]["head_distance"] else None
            else:
                label, ratio, ahead, _, _ = signals(rows, base, later, exp.budget)
                if mode == "depth_io":
                    view = 4 if ahead and ratio >= F(1, 8) else None
                elif mode in {"rank_only", "rank_near"}:
                    view = None
                    if label in {"capacity", "transfer_queue", "reuse", "overlap"}:
                        score = scorer(rows[base])
                        best_other = min(scorer(d) for c, d in rows.items() if c != base)
                        allowed = 0
                        if mode == "rank_near" and label != "overlap":
                            allowed = min(30, max(d["serial_io_ms"] for d in rows.values()) / 4)
                        if best_other < score or (allowed and best_other - score <= allowed):
                            view = 4 if ahead or later else "current"
                else:
                    threshold = F(1, 4) if mode == "route_tight" else F(1, 8)
                    if ratio < threshold or label not in {"capacity", "transfer_queue", "reuse", "overlap"}:
                        view = None
                    elif label == "overlap" and later == 0:
                        view = "current"
                    else:
                        view = 4
            if calls is not None:
                calls[str(view)] += 1
            return view
        return route

    def tail(r):
        return p95([t - a for t, a in zip(r["completion_ms"], exp.arrivals, strict=True)])

    reference = None
    for policy in policies:
        calls = Counter()
        if policy in {"current", "next4"} | EXACT_SEARCH:
            zero = timed = opts | ({"queue_horizon": 4} if policy != "current" else {})
            if policy in EXACT_SEARCH:
                zero = timed = zero | {"screen_search": policy}
        else:
            zero = klass | {"screen_selector": selector(policy, calls)}
            timed = klass | {"screen_selector": selector(policy, None)}
        run = exp.run("snapshot-screened", **zero)
        if policy == "next4":
            reference = run
        if policy in EXACT_SEARCH:
            assert reference and all(run[x] == reference[x] for x in ("boundaries_tokens", "completion_ms")), (key, policy)
        fixed = exp.run("fixed", boundaries=run["boundaries_tokens"])
        assert fixed["completion_ms"] == run["completion_ms"]
        wall = exp.run("snapshot-screened", **timed, inject_measured=True) if wall_base is not None else None
        results.writerow(dict(**key, policy=policy, requests=len(exp.requests), replays=run["decision_replays"],
            forecast_events=run["decision_events"], pruned=run.get("pruned_rollouts", 0),
            current_calls=calls["current"], queue_calls=calls["4"],
            mean_gain_ms=float(fixture["response_mean_ms"] - run["response_mean_ms"]),
            p95_gain_ms=float(tail(fixture) - tail(run)),
            timed_gain_ms=float(wall_base["response_mean_ms"] - wall["response_mean_ms"]) if wall else "",
            timed_p95_ms=float(tail(wall_base) - tail(wall)) if wall else "",
            decision_cpu_ms=sum(ns for _, ns in wall["decision_profile_ns"]) / 1e6 if wall else "",
            baseline_cpu_ms=sum(ns for _, ns in wall_base["decision_profile_ns"]) / 1e6 if wall else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", type=int, nargs="+", default=[18, 24])
    p.add_argument("--windows", type=int, default=6)
    p.add_argument("--rates", type=F, nargs="+", default=[F(1), F(8), F(16), F(32)])
    p.add_argument("--max-prompt", type=int, choices=(4096, 8192), default=4096)
    p.add_argument("--path", choices=("all", "ssd"), default="all")
    p.add_argument("--trace", nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--zero-only", action="store_true")
    p.add_argument("--no-snapshots", action="store_true", help="omit the fixed-state diagnostic")
    p.add_argument("--policies", nargs="+", choices=("current","next4","depth","depth_io","route","route_tight",
                                                    "rank_only","rank_near", *sorted(EXACT_SEARCH)),
                   default=("current","next4","depth","depth_io","route","route_tight"))
    p.add_argument("--grid", type=Path, default=ROOT / "docs/evidence/prefill-vulkan-8k.json",
                   help="Continuation cost grid; default reproduces the archived experiment")
    p.add_argument("--io-profile", type=Path, default=ROOT / "docs/evidence/load-staged-ssd-profile.json",
                   help="Transfer cost profile; default reproduces the archived experiment")
    args = p.parse_args()
    if args.windows < 1 or min(args.cohort) < 0 or min(args.rates) <= 0:
        p.error("Invalid window or cost configuration")
    if any(x in EXACT_SEARCH for x in args.policies) and args.policies[0] != "next4":
        p.error("Run next4 first when comparing optimized variants")
    fixture = read(ROOT / "fixtures/phase1-hand-trace.json")
    model = read(ROOT / "configs/qwen3.5-4b.json")["models"][0]
    grid = read(args.grid)
    io = read(args.io_profile)["cost"]
    overlap = read(ROOT / "configs/overlap-observed-upper.json")
    traces = [r["name"] for r in read(ROOT / "data/bailian-manifest.json")["files"]
              if not args.trace or r["name"] in args.trace]
    if not traces:
        p.error("Unknown trace")
    keys = ("cohort", "trace", "window", "placement", "spacing", "budget", "io_scale", "path", "max_prompt")
    with args.out.open("w", encoding="utf-8", newline="") as out, args.out.with_suffix(".snapshots.csv").open(
            "w", encoding="utf-8", newline="") as observations:
        results = csv.DictWriter(out, (*keys, "policy", "requests", "replays", "forecast_events", "pruned", "current_calls",
            "queue_calls", "mean_gain_ms", "p95_gain_ms", "timed_gain_ms", "timed_p95_ms", "decision_cpu_ms",
            "baseline_cpu_ms"), lineterminator="\n")
        snapshots = csv.DictWriter(observations, (*keys, "label", "decisions", "gains", "gains_gt5", "sum_gain",
            "relative", "ahead", "later", "slack_ms", "debt_ms", "routing_captures_gt5", "depth_captures_gt5"),
            lineterminator="\n")
        results.writeheader()
        snapshots.writeheader()
        for cohort, trace in product(args.cohort, traces):
            with (ROOT / "data/raw/bailian" / trace).open(encoding="utf-8") as stream:
                for _ in islice(stream, 352 * cohort):
                    pass
                for window in range(cohort, cohort + args.windows):
                    rows = [json.loads(line, parse_float=str) for line in islice(stream, 352)]
                    if len(rows) != 352:
                        raise ValueError(f"Incomplete window {trace}/{window}")
                    for placement, spacing in product(("split", "ram", "ssd"), (F(1), F(1, 4))):
                        data = build(rows, 256, fixture, model, spacing, placement, args.max_prompt)
                        for rate, budget in product(args.rates, (128, 384)):
                            cost = dict(fixture["cost"], recompute=grid)
                            for path, cfg in io.items():
                                factor = rate if path == "ssd_to_device" or (
                                    path == "ram_to_device" and args.path == "all") else 1
                                cost[path] = dict(startup_ms=str(F(cfg["startup_ms"]) * factor),
                                                  bytes_per_ms=str(F(cfg["bytes_per_ms"]) / factor))
                            exp = Experiment(dict(data, cost=cost, overlap_slowdown=overlap), model, budget)
                            key = dict(zip(keys, (cohort, trace, window, placement, float(spacing), budget, rate,
                                                  args.path, args.max_prompt), strict=True))
                            run_case(exp, key, results, None if args.no_snapshots else snapshots, args.zero_only,
                                     args.policies)
                    out.flush()
                    observations.flush()


if __name__ == "__main__":
    main()
