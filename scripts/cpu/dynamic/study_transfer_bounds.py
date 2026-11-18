"""Audit a decision-time transfer lower bound and time exact window selection."""

import argparse
from copy import deepcopy
import csv
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
from random import Random
from statistics import mean, median

from scripts.cpu.dynamic.dynamic_cpu_inputs import FIXTURE, MODEL, ROOT, read, sha256, verify_traces
from scripts.cpu.replay.study_bailian import Experiment
from scripts.cpu.dynamic.study_dynamic_cpu_holdout import configurations, response, timing_cost, write_csv
from scripts.cpu.dynamic.study_entry_repeatability import run


METRICS = {
    "decision_replays", "decision_events", "pruned_rollouts", "decision_profile_ns",
    "transfer_bound_checks", "transfer_bound_prunes", "transfer_bound_audits",
}


def execution(run):
    return {k: v for k, v in run.items() if k not in METRICS}


def inspect(exp):
    original = run(exp, "entry-fast")
    candidate = run(exp, "entry-transfer")
    audited = run(exp, "entry-transfer-audit")
    assert execution(original) == execution(candidate) == execution(audited)
    if "transfer_bound_checks" in candidate:
        assert audited["transfer_bound_checks"] == audited["transfer_bound_audits"]
        assert candidate["transfer_bound_prunes"] == audited["transfer_bound_prunes"]
    return original, candidate


def synthetic(seed):
    rng = Random(seed)
    fixture, model = read(FIXTURE), read(MODEL)["models"][0]
    records = []
    for shared, overlap, budget, rate, arrival in product(
            (False, True), (False, True), (0, 70, 128, 160, 192), (1, 4, 16, 32), (1, 2, 15, 29, 35)):
        trace = deepcopy(fixture)
        if shared:
            trace["requests"][2] = dict(deepcopy(trace["requests"][1]), id="C")
            trace["cost"]["device_copy"] = {"startup_ms": "0", "bytes_per_ms": 100_000_000}
        trace["requests"][2]["arrival_ms"] = arrival
        trace["requests"][1]["arrival_ms"] = F(rng.randrange(1, 7), 3)
        trace["cost"]["ssd_to_device"]["bytes_per_ms"] = str(
            F(trace["cost"]["ssd_to_device"]["bytes_per_ms"]) / rate)
        if overlap:
            trace["overlap_slowdown"] = {"ssd_to_device": {"compute": "5/4", "transfer": "3/2"}}
        exp = Experiment(trace, model, budget)
        old, candidate = inspect(exp)
        exact = exp.offline()
        assert candidate["response_sum_ms"] >= exact["response_sum_ms"]
        records.append(dict(shared=int(shared), overlap=int(overlap), budget=budget, rate=rate,
                            b_arrival=str(trace["requests"][1]["arrival_ms"]), c_arrival=arrival,
                            checks=candidate.get("transfer_bound_checks", 0),
                            prunes=candidate.get("transfer_bound_prunes", 0),
                            old_replays=old["decision_replays"], new_replays=candidate["decision_replays"]))
    return records


def benchmark(exp, key, repeats, rng):
    old, new = inspect(exp)
    for _ in range(2):
        run(exp, "entry-fast", profile=True)
        run(exp, "entry-transfer", profile=True)
    data = []
    for phase in ("profile", "inject"):
        orders = [("entry-fast", "entry-transfer"), ("entry-transfer", "entry-fast")] * (repeats // 2)
        rng.shuffle(orders)
        for trial, order in enumerate(orders):
            results = {mode: run(exp, mode, inject=phase == "inject", profile=phase == "profile")
                       for mode in order}
            a, b = results["entry-fast"], results["entry-transfer"]
            if phase == "profile":
                assert execution(a) == execution(old) == execution(b)
            va, vb = response(exp, a), response(exp, b)
            data.append(dict(
                **key, phase=phase, trial=trial, first=order[0],
                old_decision_ms=timing_cost(a), new_decision_ms=timing_cost(b),
                decision_saving_ms=timing_cost(a) - timing_cost(b),
                mean_saving_ms=float(a["response_mean_ms"] - b["response_mean_ms"]),
                worst_request_loss_ms=float(max(0, *(v-u for u,v in zip(va, vb, strict=True)))),
            ))
    return data, dict(
        **key, old_replays=old["decision_replays"], new_replays=new["decision_replays"],
        old_events=old["decision_events"], new_events=new["decision_events"],
        bound_checks=new["transfer_bound_checks"], bound_prunes=new["transfer_bound_prunes"],
        old_decision_ms=median(d["old_decision_ms"] for d in data if d["phase"] == "profile"),
        new_decision_ms=median(d["new_decision_ms"] for d in data if d["phase"] == "profile"),
        paired_saving_ms=median(d["decision_saving_ms"] for d in data if d["phase"] == "profile"),
        net_saving_ms=median(d["mean_saving_ms"] for d in data if d["phase"] == "inject"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=ROOT / "data/raw/bailian")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--holdout", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2 or args.repeats % 2:
        parser.error("Use a positive even repeat count")
    verify_traces(args.trace_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    synth = synthetic(args.seed)
    write_csv(args.out.with_name(args.out.stem + "-synthetic.csv"), synth)
    print(f"Audited {len(synth)} synthetic configurations", flush=True)
    records, paired = [], []
    rng = Random(args.seed)
    for number, (key, exp) in enumerate(configurations(args.trace_dir, calibration=not args.holdout), 1):
        key = dict(zip(("path", "scale", "trace", "window", "placement", "spacing", "budget"), key))
        key["spacing"] = str(key["spacing"])
        repeats, summary = benchmark(exp, key, args.repeats, rng)
        paired.extend(repeats)
        records.append(summary)
        if number % 24 == 0:
            print(f"Timed {number} cases", flush=True)
    write_csv(args.out, records)
    write_csv(args.out.with_name(args.out.stem + "-paired.csv"), paired)
    args.out.with_suffix(".json").write_text(json.dumps(dict(
        parameters=dict(seed=args.seed, repeats=args.repeats, holdout=args.holdout),
        cases=len(records), synthetic_cases=len(synth),
        source_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in
                       (Path(__file__).resolve(), ROOT / "src/restore_edge_select/simulator.py")},
    ), indent=2) + "\n", encoding="utf-8")
    print({field: mean(r[field] for r in records) for field in
           ("old_decision_ms", "new_decision_ms", "paired_saving_ms", "net_saving_ms")}, flush=True)


if __name__ == "__main__":
    main()
