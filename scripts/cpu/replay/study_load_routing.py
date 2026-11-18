"""Run paired recovery-boundary comparisons on Bailian trace windows."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
from fractions import Fraction as F
from itertools import islice, product
import json
from pathlib import Path
from statistics import mean

from scripts.cpu.replay.study_bailian import Experiment, ROOT, build, p95, read
from scripts.cpu.replay.study_boundary_factors import predict
from restore_edge_select.adaptive import AdaptiveConfig


CLASSES = ("idle", "light_io", "capacity", "transfer_queue", "reuse", "overlap", "uncoupled")
EXACT_SEARCH = {"entry"}
ADAPTIVE = {"adaptive", "adaptive-audit", "adaptive-strict", "adaptive-screened", "adaptive-screened-audit",
            "adaptive-fixed2", "adaptive-exhaustive", "adaptive-mean", "adaptive-mean-audit"}
MIXED_PROJECTED = "mixed-projected-entry"
MIXED_ESTIMATE = "mixed-estimate"
MIXED_REUSE_ESTIMATE = "mixed-reuse-estimate"
REUSE_CROSS_TIER_FALLBACK = "reuse-cross-tier-fallback"
REUSE_CROSS_TIER_GUARD = "reuse-cross-tier-guard"
REUSE_PREFETCH_FALLBACK = "reuse-prefetch-fallback"

ESTIMATE_POLICIES = {
    MIXED_ESTIMATE: "estimate",
    MIXED_REUSE_ESTIMATE: "estimate",
    REUSE_CROSS_TIER_FALLBACK: "estimate-cross-tier",
    REUSE_CROSS_TIER_GUARD: "estimate-cross-tier-guard",
    REUSE_PREFETCH_FALLBACK: "estimate-prefetch-fallback",
}
PRESSURE_POLICIES = {
    MIXED_PROJECTED: False,
    MIXED_ESTIMATE: False,
    MIXED_REUSE_ESTIMATE: True,
    REUSE_CROSS_TIER_FALLBACK: True,
    REUSE_CROSS_TIER_GUARD: True,
    REUSE_PREFETCH_FALLBACK: True,
}


def pressure_view(rows, base, *, allow_ready_reuse):
    """Use incremental transfer work, reuse, and the current queue state."""
    candidates = [row for boundary, row in rows.items() if boundary]
    if not rows[base]["head_distance"] or not candidates:
        return None
    tiers = {row["checkpoint_tier"] for row in candidates}
    if not tiers <= {"RAM", "SSD"}:
        return None
    fallback_score = predict("fifo_snapshot", rows[base])
    if allow_ready_reuse and any(
        row["prefetch_fits"] and row["incremental_ram_ms"] == row["incremental_ssd_ms"] == 0
        and (row["ready_bytes"] or row["shared_reservation_bytes"] or row["active_new_kv_bytes"])
        and predict("fifo_snapshot", row) < fallback_score for row in candidates
    ):
        return 4
    recompute = rows[0]["serial_compute_ms"]
    for row in candidates:
        path = "incremental_ssd_ms" if row["checkpoint_tier"] == "SSD" else "incremental_ram_ms"
        transfer = row[path]
        if not row["prefetch_fits"] or transfer <= 0 or transfer < recompute - row["serial_compute_ms"]:
            continue
        if len(tiers) > 1 and predict("fifo_snapshot", row) >= fallback_score:
            continue
        return 4
    return None


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


REQUEST_COLUMNS = (
    "baseline_mean_ms", "baseline_p95_ms", "boundary_changes", "requests_slower",
    "requests_slower_gt10ms", "max_request_loss_ms", "timed_requests_slower_gt10ms",
    "timed_max_request_loss_ms",
)


def run_case(exp, key, results, snapshots, zero_only, policies, estimate=None, request_metrics=False, capture=None):
    scorer = lambda row: predict("fifo_snapshot", row)
    opts = dict(score_candidate=scorer, screen_mode="unsettled", estimate=estimate)
    klass = dict(score_candidate=scorer, screen_mode="classified", estimate=estimate)
    fixture = exp.run("local-cost", estimate=estimate)
    wall_base = exp.run("local-cost", estimate=estimate, inject_measured=True) if not zero_only else None
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
            view = pressure_view(rows, base, allow_ready_reuse=PRESSURE_POLICIES[mode])
            if calls is not None:
                calls[str(view)] += 1
            return view
        return route

    def tail(r):
        return p95([t - a for t, a in zip(r["completion_ms"], exp.arrivals, strict=True)])

    reference = None
    route_metrics = request_metrics and any(policy in ADAPTIVE for policy in policies)
    for policy in policies:
        calls = Counter()
        engine = "snapshot-screened"
        if policy in ADAPTIVE:
            engine = "adaptive"
            limit = max(2, max(map(len, exp.eligible))) if policy == "adaptive-exhaustive" else 2
            coupled_limit = limit if policy in {"adaptive-fixed2", "adaptive-exhaustive"} else 3
            loss = None if "-mean" in policy else F(0) if policy == "adaptive-strict" else F(10)
            options = AdaptiveConfig(audit=policy.endswith("-audit"), screen="screened" in policy,
                                     loss_ms=loss, light_replays=limit, coupled_replays=coupled_limit)
            zero = timed = {"estimate": estimate, "adaptive": options}
        elif policy in {"next4"} | EXACT_SEARCH:
            zero = timed = opts | {"queue_horizon": 4}
            if policy in EXACT_SEARCH:
                zero = timed = zero | {"screen_search": policy}

        else:
            zero = klass | {"screen_selector": selector(policy, calls)}
            timed = klass | {"screen_selector": selector(policy, None)}
            for options in (zero, timed):
                options["screen_search"] = ESTIMATE_POLICIES.get(policy, "entry")
        run = exp.run(engine, **zero)
        if policy == "next4":
            reference = run
        if policy in EXACT_SEARCH:
            assert reference and all(run[x] == reference[x] for x in ("boundaries_tokens", "completion_ms")), (key, policy)
        fixed = exp.run("fixed", boundaries=run["boundaries_tokens"])
        assert fixed["completion_ms"] == run["completion_ms"]
        if capture is not None:
            capture.setdefault("local-cost", fixture)
            capture[policy] = run
        wall = exp.run(engine, **timed, inject_measured=True) if wall_base is not None else None
        extra = {}
        if request_metrics:
            def losses(base, result):
                return [after - before for before, after in
                        zip(base["completion_ms"], result["completion_ms"], strict=True)]

            delta = losses(fixture, run)
            timed_delta = losses(wall_base, wall) if wall else []
            extra = dict(baseline_mean_ms=float(fixture["response_mean_ms"]),
                         baseline_p95_ms=float(tail(fixture)),
                         boundary_changes=sum(a != b for a, b in zip(fixture["boundaries_tokens"],
                                                                      run["boundaries_tokens"], strict=True)),
                         requests_slower=sum(x > 0 for x in delta),
                         requests_slower_gt10ms=sum(x > 10 for x in delta),
                         max_request_loss_ms=float(max(0, *delta)),
                         timed_requests_slower_gt10ms=sum(x > 10 for x in timed_delta) if wall else "",
                         timed_max_request_loss_ms=float(max(0, *timed_delta)) if wall else "")
        if route_metrics:
            extra["adaptive_routes"] = json.dumps(run.get("adaptive_routes", {}), default=float, sort_keys=True)
        results.writerow(dict(**key, policy=policy, requests=len(exp.requests), replays=run["decision_replays"],
            forecast_events=run["decision_events"], pruned=run.get("pruned_rollouts", 0),
            current_calls=calls["current"], queue_calls=calls["4"],
            mean_gain_ms=float(fixture["response_mean_ms"] - run["response_mean_ms"]),
            p95_gain_ms=float(tail(fixture) - tail(run)),
            timed_gain_ms=float(wall_base["response_mean_ms"] - wall["response_mean_ms"]) if wall else "",
            timed_p95_ms=float(tail(wall_base) - tail(wall)) if wall else "",
            decision_cpu_ms=sum(ns for _, ns in wall["decision_profile_ns"]) / 1e6 if wall else "",
            baseline_cpu_ms=sum(ns for _, ns in wall_base["decision_profile_ns"]) / 1e6 if wall else "", **extra))


def write_metadata(args):
    """Keep the exact input configurations and output hashes next to each run."""
    def descriptor(path):
        path = path.resolve()
        try:
            name = path.relative_to(ROOT).as_posix()
        except ValueError:
            name = str(path)
        return dict(path=name, sha256=sha256(path.read_bytes()).hexdigest())

    inputs = {}
    for label, path in {
        "fixture": ROOT / "fixtures/phase1-hand-trace.json",
        "model": ROOT / "configs/qwen3.5-4b.json",
        "trace_manifest": ROOT / "data/bailian-manifest.json",
        "grid": args.grid, "io_profile": args.io_profile, "overlap_profile": args.overlap_profile,
    }.items():
        inputs[label] = dict(descriptor(path), content=read(path))
    metadata = dict(
        schema_version=1,
        parameters=vars(args),
        inputs=inputs,
        outputs={"results": descriptor(args.out),
                 "snapshots": descriptor(args.out.with_suffix(".snapshots.csv"))},
    )
    args.out.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", type=int, nargs="+", default=[109])
    p.add_argument("--windows", type=int, default=3)
    p.add_argument("--rates", type=F, nargs="+", default=[F(n) for n in (1, 8, 11, 12, 16, 32)])
    p.add_argument("--max-prompt", type=int, default=3584,
                   help="exclude evaluated prompts above this length; use a limit supported by the cost grid")
    p.add_argument("--path", choices=("all", "ssd"), default="ssd")
    p.add_argument("--trace", nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--zero-only", action="store_true")
    p.add_argument("--no-snapshots", action="store_true", help="omit the fixed-state diagnostic")
    p.add_argument("--request-metrics", action="store_true", help="append paired request regression metrics")
    p.add_argument("--all-checkpoints", action="store_true", help="include every history-matched 512-token boundary")
    p.add_argument("--policies", nargs="+", choices=("next4", *PRESSURE_POLICIES,
                                                    *sorted(EXACT_SEARCH | ADAPTIVE)),
                   default=("next4", "entry", "mixed-reuse-estimate", "adaptive", "adaptive-screened"))
    p.add_argument("--grid", type=Path, default=ROOT / "docs/evidence/prefill-vulkan-4k.json",
                   help="Continuation cost grid; use the 8K table for larger prompts")
    p.add_argument("--io-profile", type=Path, default=ROOT / "docs/evidence/load-staged-ssd-profile.json",
                   help="Transfer cost profile; default reproduces the archived experiment")
    p.add_argument("--overlap-profile", type=Path, default=ROOT / "configs/overlap-observed-upper.json",
                   help="Compute and transfer slowdown factors during overlap")
    p.add_argument("--ssd-estimate-factor", type=F, default=F(1),
                   help="multiply the SSD duration seen by every policy; requires a profile without slowdown")
    args = p.parse_args()
    if args.windows < 1 or min(args.cohort) < 0 or min(args.rates) <= 0 or args.max_prompt < 1:
        p.error("Window count, rates, and max prompt must be positive; cohort must be nonnegative")
    if any(x in EXACT_SEARCH for x in args.policies) and args.policies[0] != "next4":
        p.error("Run next4 first when comparing optimized variants")
    if any(policy.endswith("-audit") for policy in args.policies) and not args.zero_only:
        p.error("Adaptive audit enumerates pruned candidates and requires --zero-only")
    fixture = read(ROOT / "fixtures/phase1-hand-trace.json")
    model = read(ROOT / "configs/qwen3.5-4b.json")["models"][0]
    grid = read(args.grid)
    io = read(args.io_profile)["cost"]
    overlap = read(args.overlap_profile)
    if args.ssd_estimate_factor <= 0:
        p.error("SSD estimate factor must be positive")
    if overlap and args.ssd_estimate_factor != 1:
        p.error("SSD estimate errors require an empty overlap slowdown profile")
    estimate = ({"ssd_to_device": args.ssd_estimate_factor}
                if args.ssd_estimate_factor != 1 else None)
    traces = [r["name"] for r in read(ROOT / "data/bailian-manifest.json")["files"]
              if not args.trace or r["name"] in args.trace]
    if not traces:
        p.error("Unknown trace")
    keys = ("cohort", "trace", "window", "placement", "spacing", "budget", "io_scale", "path", "max_prompt")
    optional = REQUEST_COLUMNS if args.request_metrics else ()
    if args.request_metrics and any(policy in ADAPTIVE for policy in args.policies):
        optional += ("adaptive_routes",)
    with args.out.open("w", encoding="utf-8", newline="") as out, args.out.with_suffix(".snapshots.csv").open(
            "w", encoding="utf-8", newline="") as observations:
        results = csv.DictWriter(out, (*keys, "policy", "requests", "replays", "forecast_events", "pruned", "current_calls",
            "queue_calls", "mean_gain_ms", "p95_gain_ms", "timed_gain_ms", "timed_p95_ms", "decision_cpu_ms",
            "baseline_cpu_ms", *optional), lineterminator="\n")
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
                        data = build(rows, 256, fixture, model, spacing, placement, args.max_prompt,
                                     all_checkpoints=args.all_checkpoints)
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
                                     args.policies, estimate=estimate, request_metrics=args.request_metrics)
                    out.flush()
                    observations.flush()
    write_metadata(args)


if __name__ == "__main__":
    main()
