"""Calibrate a state gate, then measure a frozen policy on unseen windows and costs."""

import argparse
import csv
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
from statistics import mean

from dynamic_cpu_inputs import (
    ARCHIVE, FIXTURE, GRID, IO, MANIFEST, MODEL, OVERLAP, ROOT, TRACES,
    archive_rows, make_experiment, read, sha256, verify_traces, window_data,
)
from study_bailian import p95
from study_boundary_factors import predict
from restore_edge_select.adaptive import AdaptiveConfig
from restore_edge_select.dynamic_cpu import StateGate


CALIBRATION_WINDOWS = {"ssd": (108, 112)}
HOLDOUT_WINDOWS = {"ssd": (109, 110, 111), "all": (120, 121)}
CALIBRATION_RATES = (1, 11, 12)
HOLDOUT_RATES = (1, 11, 12, 16)
LAYOUTS = tuple(product(("split", "ssd"), (F(1), F(1, 4))))
BUDGETS = (128, 384)
MARGINS = (F(1, 4), F(2), F(10), F(50))
HARDWARE_CUTOFFS = (1, 11, 12, 13)
TOL = 1e-7


def configurations(directory, calibration):
    fixture, model = read(FIXTURE), read(MODEL)["models"][0]
    windows = CALIBRATION_WINDOWS if calibration else HOLDOUT_WINDOWS
    rates = CALIBRATION_RATES if calibration else HOLDOUT_RATES
    for path, selected in windows.items():
        for trace in TRACES:
            for window, placement, spacing, data in window_data(
                    directory, trace, selected, fixture, model, LAYOUTS):
                for scale, budget in product(rates, BUDGETS):
                    yield ((path, scale, trace, window, placement, spacing, budget),
                           make_experiment(fixture, model, data, path, scale, budget))


def window_policy(exp, *, route=None, inject=False, budgeted=False, screen=False, compact=False, profile=False):
    limit = max(2, max(map(len, exp.eligible)))
    cfg = AdaptiveConfig(later=4, light_replays=2 if budgeted else limit,
                         coupled_replays=3 if budgeted else limit, loss_ms=None,
                         route=route, screen=screen, compact=compact)
    return exp.run("adaptive", adaptive=cfg, inject_measured=inject, profile_decisions=profile)


def entry_policy(exp, inject, profile=False, search_mode="entry"):
    return exp.run("snapshot-screened", screen_mode="unsettled", screen_search=search_mode,
                   queue_horizon=4, score_candidate=lambda row: predict("fifo_snapshot", row),
                   inject_measured=inject, profile_decisions=profile)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def descriptors():
    return {path.relative_to(ROOT).as_posix(): sha256(path)
            for path in (ARCHIVE, FIXTURE, GRID, IO, MANIFEST, MODEL, OVERLAP)}


def calibrate(directory, target):
    totals = {label: [F(0), 0] for label in (
        *(f"{name}:{m}" for name in ("dynamic", "static") for m in MARGINS),
        *(f"hardware:{cutoff}" for cutoff in HARDWARE_CUTOFFS),
    )}
    cases = 0
    for key, exp in configurations(directory, calibration=True):
        cases += 1
        base = exp.run("local-cost")
        fixed = window_policy(exp)
        modes = {
            **{f"{name}:{m}": window_policy(exp, route=StateGate(m, state), budgeted=True)
               for name, state in (("dynamic", True), ("static", False)) for m in MARGINS},
            **{f"hardware:{cutoff}": fixed if key[1] >= cutoff else base for cutoff in HARDWARE_CUTOFFS},
        }
        for label, run in modes.items():
            totals[label][0] += base["response_mean_ms"] - run["response_mean_ms"]
            totals[label][1] += run.get("decision_replays", 0)

    def select(prefix):
        options = [label for label in totals if label.startswith(prefix + ":")]
        # Maximize average improvement, break ties with fewer forecasts
        # and then a larger threshold.
        return max(options, key=lambda k: (totals[k][0], -totals[k][1], F(k.split(":")[1]))).split(":")[1]

    frozen = dict(
        inputs_sha256=descriptors(), calibration_windows=CALIBRATION_WINDOWS, calibration_rates=CALIBRATION_RATES,
        holdout_windows=HOLDOUT_WINDOWS, holdout_rates=HOLDOUT_RATES, traces=TRACES,
        layouts=[(p, str(s)) for p, s in LAYOUTS], budgets_mib=BUDGETS, cases=cases,
        thresholds={name: select(name) for name in ("dynamic", "static", "hardware")},
        calibration_scores={label: dict(mean_gain_ms=float(gain / cases), replays=replays)
                            for label, (gain, replays) in totals.items()},
    )
    target.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen on {cases} calibration cases: {frozen['thresholds']}", flush=True)


def timing_cost(run):
    return sum(ns for _, ns in run["decision_profile_ns"]) / 1e6


def response(exp, run):
    return [finish - arrival for finish, arrival in zip(run["completion_ms"], exp.arrivals, strict=True)]


def percent_interval(base, result, lo, hi):
    gain = base - result
    if base - lo <= TOL:
        return "", "", "no-headroom" if abs(gain) <= TOL else "no-headroom-regression"
    if base - hi <= TOL:
        return "", "", "possibly-no-headroom"
    ends = sorted((100 * gain / (base - lo), 100 * gain / (base - hi)))
    return *ends, "ratio"


def holdout(directory, frozen, target):
    if frozen["inputs_sha256"] != descriptors():
        raise ValueError("Frozen input hashes do not match the current cost model or oracle archive")
    margins = frozen["thresholds"]
    gate = StateGate(F(margins["dynamic"]))
    gates = {"dynamic": gate, "static": StateGate(F(margins["static"]), use_state=False),
             "static-same-margin": StateGate(gate.margin_ms, use_state=False)}
    archived = archive_rows()
    records = []
    for key, exp in configurations(directory, calibration=False):
        path, scale, trace, window, placement, spacing, budget = key
        reference = archived[key]
        name = "fixed-window" if scale >= int(margins["hardware"]) else "local-cost"
        runs = {}
        for mode in ("local-cost", "fixed-window", "entry", "adaptive-screened", *gates):
            def run(inject):
                if mode == "local-cost":
                    return exp.run("local-cost", inject_measured=inject)
                if mode == "entry":
                    return entry_policy(exp, inject)
                return window_policy(exp, route=gates.get(mode),
                                     inject=inject, budgeted=mode != "fixed-window",
                                     screen=mode == "adaptive-screened")
            zero, timed = run(False), run(True)
            # A separately replayed fixed plan must reproduce zero-cost timings.
            assert exp.run("fixed", boundaries=zero["boundaries_tokens"])["completion_ms"] == zero["completion_ms"]
            runs[mode] = (zero, timed)
        runs["hardware"] = runs[name]
        local, local_timed = runs["local-cost"]
        base = float(local["response_mean_ms"])
        assert abs(base - float(reference["baseline_ms"])) <= TOL, (key, base, reference["baseline_ms"])
        lo = float(reference["optimum_lower_ms"])
        hi = min(float(reference["optimum_upper_ms"]),
                 *(float(r[0]["response_mean_ms"]) for r in runs.values()))
        assert lo <= hi + TOL, key
        if reference["exact"] == "1":
            assert abs(lo - hi) <= TOL, key
        baseline_end = response(exp, local)
        timed_end = response(exp, local_timed)
        for mode, (run, timed) in runs.items():
            values, timed_values = response(exp, run), response(exp, timed)
            losses = [b - a for a, b in zip(timed_end, timed_values, strict=True)]
            value = float(run["response_mean_ms"])
            ratio_left, ratio_right, ratio_status = percent_interval(base, value, lo, hi)
            routes = run.get("adaptive_routes", {})
            records.append(dict(
                path=path, scale=scale, trace=trace, window=window, placement=placement,
                spacing=str(spacing), budget=budget, policy=mode, requests=len(values),
                mean_ms=value, p95_ms=float(p95(values)),
                mean_gain_ms=base - value, decision_ms=timing_cost(timed),
                timed_mean_ms=float(timed["response_mean_ms"]),
                net_gain_ms=float(local_timed["response_mean_ms"] - timed["response_mean_ms"]),
                timed_p95_ms=float(p95(timed_values)), timed_p95_gain_ms=float(p95(timed_end) - p95(timed_values)),
                max_request_loss_ms=float(max(0, *(b-a for a,b in zip(baseline_end, values, strict=True)))),
                timed_max_loss_ms=float(max(0, *losses)), timed_losses_gt10=sum(x > 10 for x in losses),
                replays=run.get("decision_replays", 0), timed_replays=timed.get("decision_replays", 0),
                skipped=routes.get("gate-local", {}).get("decisions", 0),
                direct=routes.get("gate-direct", {}).get("decisions", 0),
                searched=routes.get("gate-search", {}).get("decisions", 0),
                optimum_lower_ms=lo, optimum_upper_ms=hi,
                exact=int(reference["exact"] == "1"), captured_lower_pct=ratio_left,
                captured_upper_pct=ratio_right, captured_status=ratio_status,
            ))
        if len(records) % (48 * len(runs)) == 0:
            print(f"Holdout: {len(records) // len(runs)} complete configurations", flush=True)
    write_csv(target, records)
    summary = []
    for path, scale, mode in product(HOLDOUT_WINDOWS, HOLDOUT_RATES, runs):
        group = [r for r in records if (r["path"], r["scale"], r["policy"]) == (path, scale, mode)]
        avg = {key: mean(r[key] for r in group) for key in (
            "mean_ms", "mean_gain_ms", "p95_ms", "decision_ms", "timed_mean_ms", "net_gain_ms",
            "timed_p95_ms", "timed_p95_gain_ms", "optimum_lower_ms", "optimum_upper_ms",
        )}
        base = avg["mean_ms"] + avg["mean_gain_ms"]
        a, b, status = percent_interval(base, avg["mean_ms"], avg["optimum_lower_ms"], avg["optimum_upper_ms"])
        summary.append(dict(
            path=path, scale=scale, policy=mode, cases=len(group), baseline_ms=base, **avg,
            exact_cases=sum(r["exact"] for r in group),
            total_replays=sum(r["replays"] for r in group),
            gate_direct=sum(r["direct"] for r in group), gate_searched=sum(r["searched"] for r in group),
            total_requests=sum(r["requests"] for r in group),
            timed_losses_gt10=sum(r["timed_losses_gt10"] for r in group),
            worst_timed_loss_ms=max(r["timed_max_loss_ms"] for r in group),
            captured_lower_pct=a, captured_upper_pct=b, captured_status=status,
        ))
    write_csv(target.with_name(target.stem + "-summary.csv"), summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=ROOT / "data/raw/bailian")
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    verify_traces(args.trace_dir)
    if args.out is None:
        args.frozen.parent.mkdir(parents=True, exist_ok=True)
        calibrate(args.trace_dir, args.frozen)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        holdout(args.trace_dir, read(args.frozen), args.out)


if __name__ == "__main__":
    main()
