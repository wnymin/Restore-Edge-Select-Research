"""Rebuild the R04-CPU-02 trace summary from raw replay CSVs."""
import argparse
import csv
import json
import gzip
from pathlib import Path


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def summarize(records, policy, scale, gain_key):
    selected = [r for r in records if r["policy"] == policy and r["io_scale"] == str(scale)]
    if len(selected) != 36:
        raise ValueError(f"expected 36 configurations for {policy}/{scale}, got {len(selected)}")
    timed = gain_key == "timed_gain_ms"
    tail_count = "timed_requests_slower_gt10ms" if timed else "requests_slower_gt10ms"
    tail_max = "timed_max_request_loss_ms" if timed else "max_request_loss_ms"
    p95 = "timed_p95_ms" if timed else "p95_gain_ms"
    return {
        "configurations": len(selected),
        "requests": sum(int(r["requests"]) for r in selected),
        "mean_gain_ms_per_config": sum(float(r[gain_key] or 0) for r in selected) / len(selected),
        "boundary_changes": sum(int(r["boundary_changes"]) for r in selected),
        "requests_slower_gt10ms": sum(int(r[tail_count]) for r in selected),
        "max_request_loss_ms": max(float(r[tail_max]) for r in selected),
        "mean_p95_gain_ms_per_config": sum(float(r[p95]) for r in selected) / len(selected),
        "mean_decision_cpu_ms_per_config": (sum(float(r["decision_cpu_ms"]) for r in selected) / len(selected)
                                            if timed else None),
        "replays": sum(int(r["replays"]) for r in selected),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero", required=True, type=Path)
    parser.add_argument("--timed", required=True, type=Path)
    parser.add_argument("--regression", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    zero, timed = rows(args.zero), rows(args.timed)
    regression = rows(args.regression) if args.regression else []
    expected_zero = 6 * 180
    expected_timed = 5 * 180
    if len(zero) != expected_zero or len(timed) != expected_timed:
        raise ValueError(f"unexpected replay row counts: {len(zero)} / {len(timed)}")
    result = {
        "schema": "r04-cpu-02-summary-v2",
        "scope": "CPU event simulation; Qwen3.5-4B; traceB windows 200-202; 36 configurations and 3360 requests per cost scale",
        "zero_overhead": [], "timed": [], "paired_policy_comparison": [], "regression_109": [],
    }
    for scale in (8, 11, 12, 16, 32):
        for policy in ("entry", "adaptive", "adaptive-no-window"):
            result["zero_overhead"].append({"scale": scale, "policy": policy,
                                             **summarize(zero, policy, scale, "mean_gain_ms")})
            result["timed"].append({"scale": scale, "policy": policy,
                                     **summarize(timed, policy, scale, "timed_gain_ms")})
        keys = ("cohort", "trace", "window", "placement", "spacing", "budget", "io_scale", "path", "max_prompt")
        def by_config(policy):
            return {tuple(r[k] for k in keys): r for r in zero
                    if r["io_scale"] == str(scale) and r["policy"] == policy}
        a, b = by_config("adaptive"), by_config("adaptive-no-window")
        if len(a) != 36 or a.keys() != b.keys():
            raise ValueError("paired policy configuration count mismatch")
        result["paired_policy_comparison"].append({
            "scale": scale, "paired_configurations": len(a),
            "different_boundaries": sum(a[k]["boundary_sha256"] != b[k]["boundary_sha256"] for k in a),
            "different_completion_sequences": sum(a[k]["completion_sha256"] != b[k]["completion_sha256"] for k in a),
            "different_mean_gain": sum(a[k]["mean_gain_ms"] != b[k]["mean_gain_ms"] for k in a),
            "different_replay_counts": sum(a[k]["replays"] != b[k]["replays"] for k in a),
        })
    if regression:
        key_fields = ("cohort", "trace", "window", "placement", "spacing", "budget", "io_scale", "path", "max_prompt")
        for policy in ("adaptive-audit", "adaptive-no-window"):
            selected = [r for r in regression if r["policy"] == policy]
            if len(selected) != 648:
                raise ValueError(f"expected 648 cohort-109 rows for {policy}; got {len(selected)}")
            comparisons = []
            other = "adaptive-no-window" if policy == "adaptive-audit" else "adaptive-audit"
            by_key = {tuple(r[k] for k in key_fields): r for r in regression if r["policy"] == other}
            mine = {tuple(r[k] for k in key_fields): r for r in selected}
            comparisons = [{"configurations": len(mine),
                "different_boundaries": sum(mine[k]["boundary_sha256"] != by_key[k]["boundary_sha256"] for k in mine),
                "different_completion_sequences": sum(mine[k]["completion_sha256"] != by_key[k]["completion_sha256"] for k in mine)}]
            result["regression_109"].append({"policy": policy, "configurations": len(selected),
                "requests": sum(int(r["requests"]) for r in selected),
                "sequence_comparison_vs_" + other: comparisons[0],
                "mean_zero_overhead_gain_ms_per_config": sum(float(r["mean_gain_ms"]) for r in selected) / len(selected),
                "requests_slower_gt10ms": sum(int(r["requests_slower_gt10ms"]) for r in selected),
                "max_request_loss_ms": max(float(r["max_request_loss_ms"]) for r in selected)})
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
