"""Summarize paired R04 policy outcomes for the R05 CPU-01 inventory."""

import argparse
import csv
import hashlib
import gzip
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def summarize(path: Path) -> dict:
    raw = path.read_bytes()
    payload = gzip.decompress(raw) if path.suffix == ".gz" else raw
    rows = list(csv.DictReader(payload.decode("utf-8-sig").splitlines()))
    paired = defaultdict(dict)
    for row in rows:
        key = (row["window"], row["placement"], row["spacing"], row["budget"], row["io_scale"])
        policy = row["policy"]
        if policy in paired[key]:
            raise ValueError(f"duplicate policy {policy} for {key}")
        paired[key][policy] = row
    if not paired:
        raise ValueError("input contains no paired rows")

    scales = sorted({key[-1] for key in paired}, key=float)
    result = {
        "schema": "r05-cpu01-r04-paired-summary-v1",
        "source": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "interpretation": "timed_gain_ms is gain over local-cost with decision-cost injection; not pairwise gain over entry",
        "scales": [],
    }
    for scale in scales:
        cases = [v for k, v in paired.items() if k[-1] == scale]
        policies = sorted(set.intersection(*(set(c) for c in cases)))
        if len(cases) != 36:
            raise ValueError(f"expected 36 configurations at scale {scale}, found {len(cases)}")
        policy_summary = []
        for policy in policies:
            gains = [float(c[policy]["timed_gain_ms"]) for c in cases]
            policy_summary.append({
                "policy": policy,
                "configurations": len(gains),
                "mean_gain_ms_per_config": sum(gains) / len(gains),
                "positive_gain_configurations_vs_local_cost": sum(x > 1e-9 for x in gains),
            })
        comparisons = []
        if {"adaptive", "entry"} <= set(policies):
            delta = [float(c["adaptive"]["timed_gain_ms"]) - float(c["entry"]["timed_gain_ms"]) for c in cases]
            comparisons.append({
                "challenger": "adaptive", "comparator": "entry",
                "challenger_wins": sum(x > 1e-9 for x in delta),
                "ties": sum(abs(x) <= 1e-9 for x in delta),
                "comparator_wins": sum(x < -1e-9 for x in delta),
                "mean_paired_gain_ms": sum(delta) / len(delta),
            })
        result["scales"].append({"io_scale": float(scale), "configurations": len(cases),
                                 "policies": policy_summary, "pairwise": comparisons})
    result["paired_configurations"] = len(paired)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "docs/evidence/r04-cpu-02-decision-overhead-injected.csv.gz")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/evidence/r05-cpu-01-r04-reproduction.json")
    args = parser.parse_args()
    result = summarize(args.input)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"summarized {result['paired_configurations']} paired configurations; wrote {args.out}")


if __name__ == "__main__":
    main()
