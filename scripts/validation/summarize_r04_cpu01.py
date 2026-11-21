"""Summarize the paired R04-CPU-01 public-trace transfer-cost ablation."""
from __future__ import annotations
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(rows, *, timed):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["policy"], row["io_scale"])].append(row)
    result = []
    for (policy, scale), items in sorted(groups.items(), key=lambda x: (float(x[0][1]), x[0][0])):
        def nums(field):
            return [float(row[field]) for row in items if row.get(field, "") != ""]
        entry = {
            "policy": policy,
            "ssd_cost_scale": scale,
            "configurations": len(items),
            "requests": sum(int(row["requests"]) for row in items),
            "boundary_changes": sum(int(row["boundary_changes"]) for row in items),
            "mean_gain_ms_per_config": sum(nums("mean_gain_ms")) / len(nums("mean_gain_ms")),
            "p95_gain_ms_per_config": sum(nums("p95_gain_ms")) / len(nums("p95_gain_ms")),
            "requests_slower_gt10ms": sum(int(row["requests_slower_gt10ms"]) for row in items),
            "max_request_loss_ms": max(nums("max_request_loss_ms"), default=0),
            "mean_replays_per_config": sum(int(row["replays"]) for row in items) / len(items),
            "mean_forecast_events_per_config": sum(int(row["forecast_events"]) for row in items) / len(items),
        }
        if timed:
            entry.update({
                "net_gain_ms_per_config": sum(nums("timed_gain_ms")) / len(nums("timed_gain_ms")),
                "net_p95_gain_ms_per_config": sum(nums("timed_p95_ms")) / len(nums("timed_p95_ms")),
                "mean_decision_cpu_ms_per_config": sum(nums("decision_cpu_ms")) / len(nums("decision_cpu_ms")),
                "requests_slower_gt10ms_after_injection": sum(
                    int(row["timed_requests_slower_gt10ms"]) for row in items),
                "max_request_loss_ms_after_injection": max(nums("timed_max_request_loss_ms"), default=0),
            })
        if policy == "adaptive":
            decisions = certified = 0
            max_gap = 0.0
            for row in items:
                for route in json.loads(row["adaptive_routes"]).values():
                    decisions += route.get("decisions", 0)
                    certified += route.get("certified", 0)
                    max_gap = max(max_gap, route.get("gap_max_ms", 0))
            entry["certificate_decisions"] = decisions
            entry["zero_gap_decisions"] = certified
            entry["max_reported_regret_gap_ms"] = max_gap
        result.append(entry)
    return result


def paired_differences(rows, a, b):
    key = lambda r: tuple(r[k] for k in (
        "cohort", "trace", "window", "placement", "spacing", "budget", "io_scale"))
    left = {key(row): row for row in rows if row["policy"] == a}
    right = {key(row): row for row in rows if row["policy"] == b}
    if left.keys() != right.keys():
        raise ValueError(f"Unpaired configurations: {a}={len(left)}, {b}={len(right)}")
    fields = ("boundary_changes", "mean_gain_ms", "p95_gain_ms", "requests_slower_gt10ms",
              "max_request_loss_ms", "replays", "forecast_events")
    # The main CSV omits full boundary vectors; compare the outcome fields it records.
    changed = 0
    for k in left:
        if any(left[k][field] != right[k][field] for field in fields if field in left[k]):
            changed += 1
    return {"policy_a": a, "policy_b": b, "paired_configurations": len(left),
            "configurations_with_different_recorded_outcomes": changed}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zero", type=Path, required=True)
    ap.add_argument("--timed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    zero, timed = read_rows(args.zero), read_rows(args.timed)
    if len(zero) != 900 or len(timed) != 720:
        raise ValueError(f"Unexpected paired rows: zero={len(zero)}, timed={len(timed)}")
    output = {
        "schema": "r04-cpu-01-summary-v1",
        "scope": "CPU event simulation; Qwen3.5-4B; public trace cohort 200, windows 200-202; all checkpoints",
        "zero_overhead": summarize(zero, timed=False),
        "decision_overhead_injected": summarize(timed, timed=True),
        "priority_only_ablation": paired_differences(zero, "adaptive", "adaptive-cost-blind"),
        "transfer_information_ablation": paired_differences(zero, "adaptive", "adaptive-no-transfer"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
