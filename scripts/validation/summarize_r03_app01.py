"""Summarize the frozen CPU evidence used by R03-APP-01.

This script only aggregates archived evidence. It does not infer per-factor closed-loop
loss when no same-input omission experiment exists.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def fixed_groups(rows, factor):
    factors = ["budget_mib", "sharing", "c_arrival_ms", "ssd_scale"]
    keys = [name for name in factors if name != factor] + ["target", "boundary"]
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return [group for group in groups.values()
            if len({row["baseline_boundary"] for row in group}) == 1
            and group[0]["baseline_boundary"] != group[0]["boundary"]]


def intervention_summary(rows):
    result = {}
    for factor in ("budget_mib", "sharing", "c_arrival_ms", "ssd_scale"):
        groups = fixed_groups(rows, factor)
        flips = [group for group in groups
                 if any(float(row["mean_gain_ms"]) > 1e-9 for row in group)
                 and any(float(row["mean_gain_ms"]) < -1e-9 for row in group)]
        result[factor] = {
            "eligible_groups": len(groups),
            "flip_groups": len(flips),
            "flip_rate": len(flips) / len(groups) if groups else None,
        }
    return result


def build_summary(evidence: Path):
    interventions = read_csv(evidence / "dynamic-cpu-interventions-synthetic-864.csv")
    outcomes = Counter(
        "positive" if float(row["mean_gain_ms"]) > 1e-9 else
        "negative" if float(row["mean_gain_ms"]) < -1e-9 else "zero"
        for row in interventions
    )
    unseen = read_csv(evidence / "dynamic-cpu-unseen-arrival.csv")
    worst = [float(row["deterministic_worst_regret_ms"]) for row in unseen]
    randomized = [float(row["pair_randomized_regret_ms"]) for row in unseen]
    holdout = read_csv(evidence / "dynamic-cpu-holdout-summary.csv")
    policy_level = [
        {key: row[key] for key in (
            "path", "scale", "policy", "cases", "mean_gain_ms", "net_gain_ms",
            "p95_ms", "timed_p95_gain_ms", "worst_timed_loss_ms", "exact_cases")}
        for row in holdout if row["policy"] in {"local-cost", "dynamic"}
    ]
    return {
        "schema": "R03-APP-01-evidence-v1",
        "method": "R03-MET-01/v1.0",
        "factor_version": "R03-INV-01/v1.0",
        "synthetic_interventions": {
            "records": len(interventions),
            "outcomes": dict(outcomes),
            "factor_groups": intervention_summary(interventions),
            "source": "dynamic-cpu-interventions-synthetic-864.csv",
        },
        "unseen_arrival": {
            "rows": len(unseen),
            "max_deterministic_worst_regret_ms": max(worst, default=0),
            "max_pair_randomized_regret_ms": max(randomized, default=0),
            "positive_configurations": sum(value > 0 for value in worst),
            "source": "dynamic-cpu-unseen-arrival.csv",
        },
        "holdout_policy_level": policy_level,
        "attribution_limits": [
            "Synthetic intervention flip rates are fixed-boundary mechanism sensitivity, not production frequencies or closed-loop policy loss.",
            "Holdout policy results are not attributable to one factor without factor-specific ablation.",
            "No per-factor L is inferred where no same-input omission/closed-loop ablation exists.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=Path("docs/evidence"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(args.evidence)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["synthetic_interventions"], ensure_ascii=False))


if __name__ == "__main__":
    main()
