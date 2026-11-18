"""Balanced ablations for exact committed-decision replay optimizations."""

import argparse
from fractions import Fraction as F
from itertools import permutations
import json
from pathlib import Path
from platform import python_version
from random import Random
from statistics import mean, median

from scripts.cpu.dynamic.dynamic_cpu_inputs import FIXTURE, MODEL, ROOT, read, sha256, verify_traces
from scripts.cpu.replay.study_bailian import Experiment, p95
from scripts.cpu.dynamic.study_dynamic_cpu_holdout import configurations, response, write_csv
from scripts.cpu.dynamic.study_entry_repeatability import compatibility, observe, run, same_result
from scripts.cpu.dynamic.study_offline_invariants import cases


POLICIES = ("entry-fast", "entry-commit", "entry-replay")


def inspect_synthetic(seed):
    model = read(MODEL)["models"][0]
    rows = []
    for number, _, _, budget, data in cases(seed, 240):
        exp = Experiment(data, model, budget)
        original = run(exp, "entry-fast")
        for mode in POLICIES[1:]:
            assert run(exp, mode) == original, (number, mode)
        rows.append(dict(case=number, requests=len(exp.requests), budget=budget,
                         window=exp.window, release=exp.release_at, exact_output_match=1))
    return rows


def measure(exp, key, repeats, rng):
    original = run(exp, "entry-fast")
    for mode in POLICIES[1:]:
        assert run(exp, mode) == original, (key, mode)
    for _ in range(2):
        for mode in POLICIES:
            run(exp, mode, profile=True)
    paired = []
    for phase in ("profile", "inject"):
        orders = list(permutations(POLICIES)) * (repeats // 6)
        rng.shuffle(orders)
        for trial, order in enumerate(orders):
            results = {mode: observe(exp, mode, phase) for mode in order}
            base, base_values = results["entry-fast"]
            for mode in POLICIES:
                result, values = results[mode]
                if phase == "profile":
                    same_result(original, result)
                old, new = response(exp, base), response(exp, result)
                paired.append(dict(
                    **key, phase=phase, trial=trial, position=order.index(mode), policy=mode,
                    decision_ms=values["decision_ms"], wall_ms=values["wall_ms"],
                    response_ms=values["response_ms"], p95_ms=values["p95_ms"],
                    decision_saving_ms=base_values["decision_ms"] - values["decision_ms"],
                    mean_saving_ms=base_values["response_ms"] - values["response_ms"],
                    p95_saving_ms=float(p95(old) - p95(new)),
                    worst_request_loss_ms=float(max(F(0), *(v-u for u, v in zip(old, new, strict=True)))),
                ))
    summaries = []
    for mode in POLICIES:
        profile = [r for r in paired if r["phase"] == "profile" and r["policy"] == mode]
        injected = [r for r in paired if r["phase"] == "inject" and r["policy"] == mode]
        summaries.append(dict(
            **key, policy=mode, requests=len(exp.requests), zero_ms=float(original["response_mean_ms"]),
            replays=original["decision_replays"], events=original["decision_events"],
            decision_ms=median(r["decision_ms"] for r in profile),
            paired_decision_saving_ms=median(r["decision_saving_ms"] for r in profile),
            wall_ms=median(r["wall_ms"] for r in profile),
            net_saving_ms=median(r["mean_saving_ms"] for r in injected),
            p95_saving_ms=median(r["p95_saving_ms"] for r in injected),
            worst_request_loss_ms=max(r["worst_request_loss_ms"] for r in injected),
        ))
    return summaries, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=ROOT / "data/raw/bailian")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 6 or args.repeats % 6:
        parser.error("Use a positive multiple of six balanced repetitions")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    synthetic = inspect_synthetic(args.seed + 1)
    compatible = compatibility(args.seed, modes=POLICIES[1:])
    write_csv(args.out.with_name(args.out.stem + "-synthetic.csv"), synthetic)
    write_csv(args.out.with_name(args.out.stem + "-compatibility.csv"), compatible)
    print(f"Validated {len(synthetic)} exact-clock and {len(compatible)} compatibility cases", flush=True)
    summaries, paired = [], []
    if not args.synthetic_only:
        verify_traces(args.trace_dir)
        rng = Random(args.seed)
        for number, (key, exp) in enumerate(configurations(args.trace_dir, calibration=not args.holdout), 1):
            key = dict(zip(("path", "rate", "trace", "window", "placement", "spacing", "budget"), key, strict=True))
            key["spacing"] = str(key["spacing"])
            summary, trials = measure(exp, key, args.repeats, rng)
            summaries.extend(summary)
            paired.extend(trials)
            if number % 24 == 0:
                print(f"Measured {number} configurations", flush=True)
        write_csv(args.out, summaries)
        write_csv(args.out.with_name(args.out.stem + "-paired.csv"), paired)
        print({mode: {k: mean(row[k] for row in summaries if row["policy"] == mode)
                      for k in ("decision_ms", "paired_decision_saving_ms", "net_saving_ms")}
               for mode in POLICIES}, flush=True)
    args.out.with_suffix(".json").write_text(json.dumps(dict(
        python=python_version(), seed=args.seed, repeats=args.repeats, holdout=args.holdout,
        synthetic_cases=len(synthetic), compatibility_cases=len(compatible), cases=len(summaries) // len(POLICIES),
        source_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in (
            Path(__file__).resolve(), ROOT / "src/restore_edge_select/simulator.py",
            ROOT / "scripts/cpu/dynamic/study_entry_repeatability.py", ROOT / "scripts/cpu/dynamic/study_offline_invariants.py",
            ROOT / "scripts/cpu/dynamic/dynamic_cpu_inputs.py", FIXTURE, MODEL)},
    ), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
