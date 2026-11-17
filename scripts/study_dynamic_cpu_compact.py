"""Compare serial search bookkeeping on a fresh CPU holdout."""

import argparse
from collections import defaultdict
from dataclasses import replace
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
from random import Random
from statistics import mean, median

from dynamic_cpu_inputs import (
    EVIDENCE, FIXTURE, MODEL, ROOT, TRACES, make_experiment, read, verify_traces, window_data,
)
from restore_edge_select.adaptive import AdaptiveConfig
from restore_edge_select.dynamic_cpu import StateGate
from restore_edge_select.offline_bounds import search
from study_bailian import p95
from study_dynamic_cpu_holdout import (
    configurations, entry_policy, percent_interval, response, timing_cost, window_policy, write_csv,
)


WINDOWS = (68, 70, 72)
FOLLOWUP_WINDOWS = (69, 71, 76)
IO_MARGIN_WINDOWS = (65, 67, 73)
ENTRY_WINDOWS = (79, 80, 81)
ENTRY_DENSE_WINDOWS = (82, 83)
ENTRY_DENSE_RATES = (12, 32)
RATES = (1, 12, 20, 32)
LAYOUTS = (("ssd", F(1)), ("split", F(1, 4)))
BUDGETS = (128, 384)
POLICIES = ("local-cost", "entry", "dynamic", "compact")
ENTRY_VARIANTS = ("entry-fast", "entry-progress")


def experiments(directory, windows, rates=RATES, *, all_checkpoints=False):
    fixture, model = read(FIXTURE), read(MODEL)["models"][0]
    for trace in TRACES:
        for window, placement, spacing, data in window_data(
                directory, trace, windows, fixture, model, LAYOUTS, all_checkpoints=all_checkpoints):
            for path, rate, budget in product(("ssd", "all"), rates, BUDGETS):
                key = dict(trace=trace, window=window, path=path, rate=rate,
                           placement=placement, spacing=str(spacing), budget=budget)
                yield key, make_experiment(fixture, model, data, path, rate, budget)


def replay(exp, gate, policy, *, inject=False, profile=False):
    if policy == "local-cost":
        return exp.run("local-cost", inject_measured=inject, profile_decisions=profile)
    if policy in ("entry", *ENTRY_VARIANTS):
        return entry_policy(exp, inject, profile, search_mode=policy)
    if policy in ("check-blocked", "check-io-margin"):
        gate = replace(gate, check_blocked_baseline=True, require_io_margin=policy == "check-io-margin")
    return window_policy(exp, route=gate, budgeted=True, compact=policy != "dynamic",
                         inject=inject, profile=profile)


def verify_equivalence(directory, gate):
    count = 0
    for key, exp in configurations(directory, calibration=False):
        assert replay(exp, gate, "dynamic") == replay(exp, gate, "compact"), key
        count += 1
    return count


def check_entry_equivalence(key, reference, runs):
    search_metrics = {"decision_replays", "decision_events", "pruned_rollouts"}
    expected = {k: v for k, v in reference.items() if k not in search_metrics}
    for mode, run in runs.items():
        observed = {k: v for k, v in run.items() if k not in search_metrics}
        assert expected == observed, (key, mode, [k for k in expected if expected[k] != observed.get(k)])


def verify_entry(directory, gate):
    groups = (configurations(directory, calibration=False), *(
        experiments(directory, windows) for windows in (WINDOWS, FOLLOWUP_WINDOWS, IO_MARGIN_WINDOWS)
    ))
    count = 0
    for group in groups:
        for key, exp in group:
            reference = replay(exp, gate, "entry")
            check_entry_equivalence(key, reference, {mode: replay(exp, gate, mode) for mode in ENTRY_VARIANTS})
            count += 1
    return count


def blocked_witness(directory, gate):
    fixture, model = read(FIXTURE), read(MODEL)["models"][0]
    trace, window, placement, spacing = TRACES[1], 70, "split", F(1, 4)
    _, _, _, data = next(window_data(directory, trace, (window,), fixture, model, ((placement, spacing),)))
    exp = make_experiment(fixture, model, data, "all", 20, 128)
    i = next(i for i, request in enumerate(exp.requests) if request["id"] == "324")
    snapshots = []

    def observe(rows, base, budget):
        if set(rows) == set(exp.eligible[i]) and all(
            row["serial_compute_ms"] == exp.recompute(i, c) for c, row in rows.items()
        ):
            snapshots.append(rows)
        return gate(rows, base, budget)

    dynamic = window_policy(exp, route=observe, budgeted=True, compact=True)
    entry = replay(exp, gate, "entry")
    checked = replay(exp, gate, "check-blocked")
    baseline, alternative = dynamic["boundaries_tokens"][i], entry["boundaries_tokens"][i]
    assert (baseline, alternative) == (512, 0) and len(snapshots) == 1
    plan = list(dynamic["boundaries_tokens"])
    plan[i] = alternative
    changed = exp.run("fixed", boundaries=plan)
    assert changed["completion_ms"] == checked["completion_ms"] == entry["completion_ms"]
    differences = [before - after for before, after in zip(
        dynamic["completion_ms"], changed["completion_ms"], strict=True)]
    return dict(trace=trace, window=window, placement=placement, spacing=str(spacing),
                path="all", rate=20, budget=128, request=exp.requests[i]["id"],
                baseline=baseline, alternative=alternative, snapshot=snapshots[0],
                earlier_gain_sum_ms=sum(differences[:i]), target_gain_ms=differences[i],
                follower_gain_sum_ms=sum(differences[i + 1:]),
                mean_gain_ms=mean(differences),
                dynamic_replays=dynamic["decision_replays"], checked_replays=checked["decision_replays"],
                entry_replays=entry["decision_replays"])


def io_margin_witness(directory, gate):
    fixture, model = read(FIXTURE), read(MODEL)["models"][0]
    _, _, _, data = next(window_data(directory, TRACES[1], (71,), fixture, model, (("split", F(1, 4)),)))
    exp = make_experiment(fixture, model, data, "ssd", 32, 128)
    i = next(i for i, request in enumerate(exp.requests) if request["id"] == "285")
    captured = []
    checked = replace(gate, check_blocked_baseline=True)

    def observe(rows, base, budget):
        if set(rows) == set(exp.eligible[i]) and all(
            row["serial_compute_ms"] == exp.recompute(i, c) for c, row in rows.items()
        ):
            captured.append(rows)
        return checked(rows, base, budget)

    wide = window_policy(exp, route=observe, budgeted=True, compact=True)
    cut = replay(exp, gate, "check-io-margin")
    plan = list(cut["boundaries_tokens"])
    assert plan[i] == 512 and wide["boundaries_tokens"][i] == 0 and len(captured) == 1
    plan[i] = 0
    assert exp.run("fixed", boundaries=plan)["completion_ms"] == wide["completion_ms"]
    return dict(trace=TRACES[1], window=71, placement="split", spacing="1/4", path="ssd", rate=32,
                budget=128, request=exp.requests[i]["id"], baseline=512, alternative=0,
                snapshot=captured[0], improvement_ms=cut["response_mean_ms"] - wide["response_mean_ms"])


def threshold_ablation(directory, gate):
    """Explain remaining differences without tuning or reporting a third policy as a holdout."""
    gate = replace(gate, check_blocked_baseline=True)
    settings = dict(loss_ms=None, route=gate, compact=True)
    for key, exp in experiments(directory, FOLLOWUP_WINDOWS):
        entry = replay(exp, gate, "entry")
        one = exp.run("adaptive", adaptive=AdaptiveConfig(**settings, gain_per_request_ms=F(1)))
        zero = exp.run("adaptive", adaptive=AdaptiveConfig(**settings, gain_per_request_ms=F(0)))
        entry_mean = entry["response_mean_ms"]
        yield dict(**key, entry_mean_ms=float(entry_mean),
                   one_gap_ms=float(one["response_mean_ms"] - entry_mean),
                   zero_gap_ms=float(zero["response_mean_ms"] - entry_mean),
                   boundaries_changed=sum(a != b for a, b in zip(
                       one["boundaries_tokens"], zero["boundaries_tokens"], strict=True)),
                   entry_replays=entry["decision_replays"], one_replays=one["decision_replays"],
                   zero_replays=zero["decision_replays"])


def collect(exp, gate, key, rng, repeats, nodes, policies):
    zero = {policy: replay(exp, gate, policy) for policy in policies}
    if "dynamic" in zero:
        assert zero["compact"] == zero["dynamic"], key
    if "entry-fast" in zero:
        check_entry_equivalence(key, zero["entry"], {mode: zero[mode] for mode in ENTRY_VARIANTS})
    raw_values = {policy: response(exp, run) for policy, run in zero.items()}
    lower, upper, plan, visited = search(exp, zero.values(), nodes)
    assert exp.run("fixed", boundaries=plan)["response_sum_ms"] == upper, key
    n = len(exp.requests)
    base = zero["local-cost"]["response_mean_ms"]

    measurements = defaultdict(list)
    for inject in (False, True):
        for _ in range(repeats):
            order = list(policies)
            rng.shuffle(order)
            runs = {policy: replay(exp, gate, policy, inject=inject, profile=not inject) for policy in order}
            if not inject:
                for policy, run in runs.items():
                    profile_only = {k: v for k, v in run.items() if k != "decision_profile_ns"}
                    assert profile_only == zero[policy], (key, policy)
                    measurements[policy, "profile_ms"].append(timing_cost(run))
                continue
            local_values = response(exp, runs["local-cost"])
            entry_mean = runs["entry"]["response_mean_ms"]
            for policy, run in runs.items():
                values = response(exp, run)
                losses = [actual - original for actual, original in zip(values, local_values, strict=True)]
                measurements[policy, "net_vs_local_ms"].append(float(
                    runs["local-cost"]["response_mean_ms"] - run["response_mean_ms"]))
                measurements[policy, "net_vs_entry_ms"].append(float(entry_mean - run["response_mean_ms"]))
                measurements[policy, "p95_gain_ms"].append(float(p95(local_values) - p95(values)))
                measurements[policy, "max_loss_ms"].append(float(max(0, *losses)))
                measurements[policy, "losses_gt10"].append(sum(loss > 10 for loss in losses))
            for policy in policies:
                paired = timing_cost(runs["entry"]) - timing_cost(runs[policy])
                measurements[policy, "injected_decision_saving_ms"].append(paired)

    for policy in policies:
        run = zero[policy]
        zero_mean = run["response_mean_ms"]
        yield dict(
            **key, policy=policy, requests=n, zero_mean_ms=float(zero_mean), local_mean_ms=float(base),
            entry_zero_difference_ms=float(zero_mean - zero["entry"]["response_mean_ms"]),
            zero_p95_ms=float(p95(raw_values[policy])), replays=run.get("decision_replays", 0),
            forecast_events=run.get("decision_events", 0),
            pruned=run.get("pruned_rollouts", 0),
            blocked_baseline_searches=run.get("adaptive_routes", {}).get("gate-blocked-baseline", {}).get(
                "decisions", 0),
            oracle_lower_ms=float(lower / n), oracle_upper_ms=float(upper / n),
            oracle_exact=int(lower == upper), oracle_nodes=visited,
            **{field: median(times) for (p, field), times in measurements.items() if p == policy},
            worst_repeat_loss_ms=max(measurements[policy, "max_loss_ms"]),
        )


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["path"], row["rate"], row["policy"]].append(row)
    for (path, rate, policy), group in groups.items():
        values = {key: mean(row[key] for row in group) for key in (
            "zero_mean_ms", "local_mean_ms", "entry_zero_difference_ms", "zero_p95_ms",
            "net_vs_local_ms", "net_vs_entry_ms", "p95_gain_ms", "profile_ms",
            "injected_decision_saving_ms", "oracle_lower_ms", "oracle_upper_ms",
        )}
        low, high, status = percent_interval(values["local_mean_ms"], values["zero_mean_ms"],
                                              values["oracle_lower_ms"], values["oracle_upper_ms"])
        yield dict(path=path, rate=rate, policy=policy, cases=len(group), **values,
                   replays=sum(r["replays"] for r in group),
                   forecast_events=sum(r["forecast_events"] for r in group),
                   pruned=sum(r["pruned"] for r in group),
                   blocked_baseline_searches=sum(r["blocked_baseline_searches"] for r in group),
                   slower_than_entry_cases=sum(r["entry_zero_difference_ms"] > 0 for r in group),
                   faster_than_entry_cases=sum(r["entry_zero_difference_ms"] < 0 for r in group),
                   timed_wins_vs_entry=sum(r["net_vs_entry_ms"] > 0 for r in group),
                   losses_gt10=sum(r["losses_gt10"] for r in group),
                   requests=sum(r["requests"] for r in group),
                   max_loss_ms=max(r["worst_repeat_loss_ms"] for r in group),
                   oracle_exact=sum(r["oracle_exact"] for r in group),
                   captured_lower_pct=low, captured_upper_pct=high, captured_status=status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=ROOT / "data/raw/bailian")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261117)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--followup", action="store_true", help="test blocked baseline checking on a second holdout")
    mode.add_argument("--threshold-ablation", action="store_true", help="explain differences in the second holdout")
    mode.add_argument("--io-margin", action="store_true", help="test prefiltering blocked baseline checks")
    mode.add_argument("--entry", action="store_true", help="validate exact entry optimizations")
    mode.add_argument("--entry-dense", action="store_true", help="test entry with every matching checkpoint")
    args = parser.parse_args()
    if args.nodes < 0 or args.repeats < 1 or args.repeats % 2 == 0:
        parser.error("Use a nonnegative oracle budget and an odd positive repeat count")
    verify_traces(args.trace_dir)
    frozen = read(EVIDENCE / "dynamic-cpu-frozen.json")
    gate = StateGate(F(frozen["thresholds"]["dynamic"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.threshold_ablation:
        write_csv(args.out, list(threshold_ablation(args.trace_dir, gate)))
        return
    equivalent_cases = 0
    if not args.entry_dense:
        equivalent_cases = (verify_entry if args.entry else verify_equivalence)(args.trace_dir, gate)
        print(f"Previous holdout: {equivalent_cases} exact matches", flush=True)
    rng = Random(args.seed)
    records = []
    rates = ENTRY_DENSE_RATES if args.entry_dense else RATES
    if args.entry or args.entry_dense:
        windows = ENTRY_DENSE_WINDOWS if args.entry_dense else ENTRY_WINDOWS
        policies = ("local-cost", "entry", *ENTRY_VARIANTS, "compact")
    elif args.io_margin:
        windows = IO_MARGIN_WINDOWS
        policies = ("local-cost", "entry", "compact", "check-blocked", "check-io-margin")
    elif args.followup:
        windows = FOLLOWUP_WINDOWS
        policies = (*POLICIES, "check-blocked")
    else:
        windows, policies = WINDOWS, POLICIES
    for i, (key, exp) in enumerate(experiments(args.trace_dir, windows, rates,
                                              all_checkpoints=args.entry_dense), 1):
        records.extend(collect(exp, gate, key, rng, args.repeats, args.nodes, policies))
        if i % 48 == 0:
            print(f"Fresh holdout: {i} paired cases", flush=True)
    write_csv(args.out, records)
    write_csv(args.out.with_name(args.out.stem + "-summary.csv"), list(summarize(records)))
    args.out.with_suffix(".json").write_text(json.dumps(dict(
        prior_holdout_equivalent=equivalent_cases, windows=windows, rates=rates, budgets=BUDGETS,
        policies=policies,
        all_checkpoints=args.entry_dense,
        layouts=[(p, str(s)) for p, s in LAYOUTS], traces=TRACES, seed=args.seed,
        repeats=args.repeats, oracle_max_nodes=args.nodes, gate_margin_ms=str(gate.margin_ms),
        prior_inputs_sha256=frozen["inputs_sha256"],
    ), indent=2) + "\n", encoding="utf-8")
    if args.followup or args.io_margin:
        witness = io_margin_witness(args.trace_dir, gate) if args.io_margin else blocked_witness(args.trace_dir, gate)
        args.out.with_suffix(".witness.json").write_text(
            json.dumps(witness, indent=2, default=float) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
