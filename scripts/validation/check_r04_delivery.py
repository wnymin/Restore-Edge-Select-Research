"""Recompute the R04 summaries and check archived evidence against the report."""
import csv
import gzip
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/evidence"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check_manifest(line):
    manifest = read_json(EVIDENCE / f"r04-cpu-{line}-manifest.json")
    for name, info in manifest["outputs"].items():
        data = (ROOT / name).read_bytes()
        assert len(data) == info["bytes"] and sha256(data).hexdigest() == info["sha256"], name
    for name, digest in manifest["sources"].items():
        assert sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


def check_summary(line):
    prefix = EVIDENCE / f"r04-cpu-{line}"
    scratch = ROOT / "outputs"
    scratch.mkdir(exist_ok=True)
    target = scratch / f"r04-check-{uuid4().hex}.json"
    command = [sys.executable, "-m", f"scripts.validation.summarize_r04_cpu{line}",
               "--zero", f"{prefix}-zero-overhead.csv.gz",
               "--timed", f"{prefix}-decision-overhead-injected.csv.gz", "--out", str(target)]
    if line == "02":
        command.extend(("--regression", str(EVIDENCE / "r04-regression-109.csv.gz")))
    try:
        subprocess.run(command, cwd=ROOT, check=True)
        summary = read_json(target)
    finally:
        target.unlink(missing_ok=True)
    assert summary == read_json(Path(f"{prefix}-summary.json")), line
    return summary


def check_archived_oracle():
    fields = ("cohort", "trace", "window", "placement", "spacing", "budget",
              "io_scale", "path", "max_prompt")

    def key(row):
        return tuple(row[name] for name in fields)

    with (EVIDENCE / "adaptive-trace-oracle.csv").open(encoding="utf-8-sig", newline="") as stream:
        oracle = {key(r): r for r in csv.DictReader(stream)
                  if r["run"] == "adaptive-holdout-109" and r["policy"] == "entry"}
    with gzip.open(EVIDENCE / "r04-regression-109.csv.gz", "rt", newline="") as stream:
        entry = {key(r): r for r in csv.DictReader(stream) if r["policy"] == "entry"}
    assert len(entry) == 648 and entry.keys() == oracle.keys()
    for k, r in entry.items():
        assert abs(float(r["baseline_mean_ms"]) - float(r["mean_gain_ms"])
                   - float(oracle[k]["strategy_ms"])) < 1e-8, k
    assert sum(int(r["exact"]) for r in oracle.values()) == 630


def main():
    for line in ("01", "02"):
        check_manifest(line)
    check_archived_oracle()
    cost = check_summary("01")
    window = check_summary("02")
    assert cost["transfer_information_ablation"]["configurations_with_different_recorded_outcomes"] == 98
    assert cost["priority_only_ablation"]["configurations_with_different_recorded_outcomes"] == 0
    pairs = window["paired_policy_comparison"]
    assert sum(r["paired_configurations"] for r in pairs) == 180
    assert all(r["different_boundaries"] == r["different_completion_sequences"] == 0 for r in pairs)
    regression, _ = window["regression_109"]
    assert regression["sequence_comparison_vs_adaptive-no-window"] == {
        "configurations": 648, "different_boundaries": 0, "different_completion_sequences": 0}
    with gzip.open(EVIDENCE / "r04-cpu-01-certificate-audit.csv.gz", "rt", newline="") as stream:
        assert sum(r["policy"] == "adaptive-audit" for r in csv.DictReader(stream)) == 72
    snapshots = read_json(EVIDENCE / "r04-fixed-snapshots.json")
    debt = snapshots["queue_debt"]
    assert len(snapshots["coverage"]) == 8 and len(debt) == 56
    assert sum(Fraction(r["queue_debt_ms"]) > 0 for r in debt) == 32
    assert sum(r["full"]["scores"] != r["ablated"]["scores"] for r in debt) == 3
    assert all(r["full"]["action"] == r["ablated"]["action"] for r in debt)
    print("R04 checked: manifests, rebuilt summaries, 828 paired configurations, and 64 snapshots.")


if __name__ == "__main__":
    main()
