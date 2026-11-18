"""Balanced repeated CPU timing, paired response differences, and compatibility checks."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
from fractions import Fraction as F
from itertools import permutations, product
import json
from pathlib import Path
from platform import python_version
from random import Random
from statistics import mean, median
from time import perf_counter_ns

from scripts.cpu.dynamic.dynamic_cpu_inputs import (
    FIXTURE, GRID, IO, MODEL, OVERLAP, ROOT, TRACES, make_experiment, read, sha256, verify_traces, window_data,
)
from scripts.cpu.replay.study_bailian import Experiment, p95
from scripts.cpu.replay.study_boundary_factors import predict
from scripts.cpu.dynamic.study_dynamic_cpu_holdout import response, timing_cost, write_csv


POLICIES = ("local-cost", "entry", "entry-fast")
CASE_KEYS = ("trace", "window", "dense", "placement", "spacing", "path", "rate", "budget")
SOURCE_PATHS = (
    FIXTURE, MODEL, GRID, IO, OVERLAP, ROOT / "data/bailian-manifest.json",
    ROOT / "src/restore_edge_select/simulator.py",
    ROOT / "scripts/cpu/dynamic/dynamic_cpu_inputs.py", Path(__file__).resolve(),
)


def run(exp, policy, *, inject=False, profile=False, **options):
    if policy == "local-cost":
        return exp.run(policy, inject_measured=inject, profile_decisions=profile)
    return exp.run("snapshot-screened", screen_mode="unsettled", queue_horizon=4,
                   score_candidate=lambda row: predict("fifo_snapshot", row), screen_search=policy,
                   inject_measured=inject, profile_decisions=profile, **options)


def same_result(original, candidate):
    metrics = {"decision_replays", "decision_events", "pruned_rollouts", "decision_profile_ns"}
    assert {k: v for k, v in original.items() if k not in metrics} == {
        k: v for k, v in candidate.items() if k not in metrics}


def compatibility(seed, *, modes=("entry-fast", "entry-progress")):
    """Exercise fallback and stationary execution with independent fixture variations."""
    fixture, model = read(FIXTURE), read(MODEL)["models"][0]
    rng = Random(seed)
    records = []
    for sharing, initial, overlap, release, budget in product(
        (False, True), (False, True), (False, True), ("start", "finish", "timer"), (0, 70, 128, 160, 192)
    ):
        trace = deepcopy(fixture)
        trace["requests"][1]["arrival_ms"] = F(rng.randrange(1, 31), 3)
        trace["requests"][2]["arrival_ms"] = F(rng.randrange(1, 91), 3)
        if sharing:
            trace["requests"][2] = dict(deepcopy(trace["requests"][1]), id="C",
                                       arrival_ms=trace["requests"][2]["arrival_ms"])
        if initial:
            trace["scheduling"]["initial_device_object_ids"] = ["B.kv1"]
        if overlap:
            trace["overlap_slowdown"] = {"ssd_to_device": {"compute": "5/4", "transfer": "3/2"}}
        if sharing:
            trace["cost"]["device_copy"] = {"startup_ms": "0", "bytes_per_ms": 100_000_000}
        trace["scheduling"].update(prefetch_release=release, prefetch_hold_ms=1 if release == "timer" else 0)
        for label in ("known", "estimate-low", "estimate-high", "actual-different",
                      "variable-rate", "staging", "per-tier"):
            case = deepcopy(trace)
            options = {}
            if label.startswith("estimate"):
                # The executor supports estimate errors separately from concurrent slowdown.
                case.pop("overlap_slowdown", None)
                options["estimate"] = {"ssd_to_device": F(1, 2) if label.endswith("low") else F(2)}
            elif label == "actual-different":
                case.pop("overlap_slowdown", None)
                case["actual_transfer_ms_by_object"] = {"B.state1": "5"}
            elif label == "variable-rate":
                case.pop("overlap_slowdown", None)
                case["io_speed_epochs"] = [(0, 1), (2, "1/2"), (9, "3/2")]
            elif label == "staging":
                case.pop("overlap_slowdown", None)
                case["staged_checkpoint_cache_mib"] = 64
            elif label == "per-tier":
                case.pop("overlap_slowdown", None)
                case["transfer_topology"] = "per-tier"
            exp = Experiment(case, model, budget)
            old = run(exp, "entry", **options)
            for mode in modes:
                got = run(exp, mode, **options)
                same_result(old, got)
                if label != "known":
                    assert got == old, label
            records.append(dict(sharing=int(sharing), initial=int(initial), overlap=int(overlap),
                                release=release, budget=budget, scenario=label, requests=len(exp.requests)))
    return records


def experiments(directory):
    fixture, model = read(FIXTURE), read(MODEL)["models"][0]
    for window, dense in ((84, False), (85, True)):
        for trace in TRACES:
            for _, placement, spacing, data in window_data(
                directory, trace, (window,), fixture, model, (("ssd", F(1)), ("split", F(1, 4))),
                all_checkpoints=dense,
            ):
                for path, rate, budget in product(("ssd", "all"), (1, 12, 20, 32), (128, 384)):
                    key = dict(trace=trace, window=window, dense=int(dense), placement=placement,
                               spacing=str(spacing), path=path, rate=rate, budget=budget)
                    yield key, make_experiment(fixture, model, data, path, rate, budget)


def observe(exp, policy, mode):
    start = perf_counter_ns()
    result = run(exp, policy, inject=mode == "inject", profile=mode == "profile")
    wall_ms = (perf_counter_ns() - start) / 1e6
    return result, dict(
        decision_ms=timing_cost(result), wall_ms=wall_ms,
        response_ms=float(result["response_mean_ms"]), p95_ms=float(p95(response(exp, result))),
        replays=result.get("decision_replays", 0), forecast_events=result.get("decision_events", 0),
    )


def measure(directory, repeats, seed, output):
    rng = Random(seed)
    cases, rounds = [], []
    keys = (*CASE_KEYS, "mode", "round", "position", "policy", "decision_ms", "wall_ms",
            "response_ms", "p95_ms", "replays", "forecast_events")
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, keys, lineterminator="\n")
        writer.writeheader()
        for number, (key, exp) in enumerate(experiments(directory), 1):
            baseline = {p: run(exp, p) for p in POLICIES}
            same_result(baseline["entry"], baseline["entry-fast"])
            for _ in range(2):
                for p in POLICIES:
                    run(exp, p, profile=True)
            measured = defaultdict(list)
            net = []
            for mode in ("profile", "inject"):
                orders = list(permutations(POLICIES)) * (repeats // 6)
                rng.shuffle(orders)
                assert all(Counter(order[pos] for order in orders) == Counter(
                    {p: repeats // 3 for p in POLICIES}) for pos in range(3))
                for repeat, order in enumerate(orders):
                    results = {}
                    for pos, policy in enumerate(order):
                        result, timing = observe(exp, policy, mode)
                        if mode == "profile":
                            same_result(baseline[policy], result)
                        results[policy] = result
                        measured[mode, policy].append(timing)
                        writer.writerow(dict(**key, mode=mode, round=repeat, position=pos,
                                             policy=policy, **timing))
                    old, fast = results["entry"], results["entry-fast"]
                    old_responses, fast_responses = response(exp, old), response(exp, fast)
                    decisions_saved = timing_cost(old) - timing_cost(fast)
                    item = dict(
                        **key, mode=mode, round=repeat, decision_saving_ms=decisions_saved,
                        mean_saving_ms=float(old["response_mean_ms"] - fast["response_mean_ms"]),
                        p95_saving_ms=float(p95(old_responses) - p95(fast_responses)),
                        requests_slower=sum(b > a for a, b in zip(old_responses, fast_responses, strict=True)),
                        max_loss_ms=float(max(0, *(b-a for a,b in zip(
                            old_responses, fast_responses, strict=True)))),
                    )
                    if mode == "inject":
                        net.append(item)
                    rounds.append(item)
            old, fast = measured["profile", "entry"], measured["profile", "entry-fast"]
            paired_savings = [row["decision_saving_ms"] for row in rounds[-2*repeats:] if row["mode"] == "profile"]
            cases.append(dict(
                **key, requests=len(exp.requests), max_candidates=max(map(len, exp.eligible)),
                zero_response_ms=float(baseline["entry"]["response_mean_ms"]),
                profile_entry_ms=median(x["decision_ms"] for x in old),
                profile_fast_ms=median(x["decision_ms"] for x in fast),
                paired_profile_saving_ms=median(paired_savings),
                net_saving_ms=median(x["mean_saving_ms"] for x in net),
                p95_saving_ms=median(x["p95_saving_ms"] for x in net),
                injected_round_wins=sum(x["mean_saving_ms"] > 0 for x in net),
                worst_request_loss_ms=max(x["max_loss_ms"] for x in net),
                replays_entry=baseline["entry"]["decision_replays"],
                replays_fast=baseline["entry-fast"]["decision_replays"],
            ))
            stream.flush()
            if number % 24 == 0:
                print(f"Completed {number} cases with {repeats} balanced rounds", flush=True)
    return cases, rounds


def interval(values):
    xs = sorted(values)
    return xs[int(0.025 * (len(xs)-1))], xs[int(0.975 * (len(xs)-1))]


def summarize(cases, seed):
    rng = Random(seed + 1)
    rows = []
    for dense, rate in product((0, 1), (1, 12, 20, 32)):
        group = [r for r in cases if (r["dense"], r["rate"]) == (dense, rate)]
        clusters = defaultdict(list)
        for r in group:
            clusters[r["trace"]].append(r)
        per_trace = [[mean(r[k] for r in cluster) for k in
                      ("paired_profile_saving_ms", "net_saving_ms")] for cluster in clusters.values()]
        draws = []
        for _ in range(4000):
            sample = rng.choices(per_trace, k=len(per_trace))
            draws.append(mean(x[1] for x in sample))
        lo, hi = interval(draws)
        rows.append(dict(
            dense=dense, rate=rate, cases=len(group), traces=len(clusters),
            profile_entry_ms=mean(r["profile_entry_ms"] for r in group),
            profile_fast_ms=mean(r["profile_fast_ms"] for r in group),
            paired_profile_saving_ms=mean(r["paired_profile_saving_ms"] for r in group),
            net_saving_ms=mean(r["net_saving_ms"] for r in group),
            p95_saving_ms=mean(r["p95_saving_ms"] for r in group),
            net_positive_cases=sum(r["net_saving_ms"] > 0 for r in group),
            profile_positive_cases=sum(r["paired_profile_saving_ms"] > 0 for r in group),
            net_trace_bootstrap_low_ms=lo, net_trace_bootstrap_high_ms=hi,
            net_min_trace_ms=min(x[1] for x in per_trace), net_max_trace_ms=max(x[1] for x in per_trace),
            worst_request_loss_ms=max(r["worst_request_loss_ms"] for r in group),
        ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=ROOT / "data/raw/bailian")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()
    if args.repeats < 6 or args.repeats % 6:
        parser.error("Repeat count must be a positive multiple of six")
    verify_traces(args.trace_dir)
    checks = compatibility(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out.with_name(args.out.stem + "-compatibility.csv"), checks)
    cases, rounds = measure(args.trace_dir, args.repeats, args.seed, args.out)
    write_csv(args.out.with_name(args.out.stem + "-paired.csv"), rounds)
    write_csv(args.out.with_name(args.out.stem + "-cases.csv"), cases)
    write_csv(args.out.with_name(args.out.stem + "-summary.csv"), summarize(cases, args.seed))
    meta = dict(seed=args.seed, repeats=args.repeats, bootstrap_draws=4000,
                bootstrap_unit="trace within window and cost group; three clusters per summary row",
                python=python_version(), compatibility_cases=len(checks), cases=len(cases),
                windows=[dict(window=84, dense=False), dict(window=85, dense=True)],
                inputs={p.relative_to(ROOT).as_posix(): sha256(p) for p in SOURCE_PATHS})
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
