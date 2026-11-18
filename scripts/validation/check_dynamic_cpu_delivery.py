"""Check the CPU delivery from archived inputs, without network or raw traces."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from scripts.cpu.dynamic.dynamic_cpu_inputs import ROOT, sha256
from scripts.cpu.dynamic.study_dynamic_cpu_holdout import descriptors
from scripts.cpu.dynamic.study_transfer_bounds import synthetic


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/dynamic-cpu-checks")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((ROOT / "docs/evidence/dynamic-cpu-frozen.json").read_text(encoding="utf-8"))
    expected = frozen["inputs_sha256"]
    actual_hashes = descriptors()
    mismatches = sorted(path for path in expected.keys() | actual_hashes.keys()
                        if expected.get(path) != actual_hashes.get(path))
    require(not mismatches, "Frozen input byte hashes differ for " + ", ".join(mismatches))

    def run(*command):
        subprocess.run([sys.executable, *map(str, command)],
                       cwd=ROOT, check=True)

    run("scripts/check_fixture.py")
    transfer = synthetic(20261120)
    archived_transfer = csv_rows(ROOT / "docs/evidence/dynamic-cpu-transfer-synthetic.csv")
    require([{k: str(v) for k, v in row.items()} for row in transfer] == archived_transfer,
            "Transfer-bound audit differs from archived synthetic cases")
    print(f"Audited {len(transfer)} transfer-bound configurations.", flush=True)
    for name, extra in (("dominance", ("--dominance",)),
                        ("combined", ("--dominance", "--resource-floor"))):
        target = args.out / f"{name}-exhaustive.csv"
        run("-m", "scripts.cpu.replay.study_adaptive_oracle", *extra, "--out", target)
        require(len(csv_rows(target)) == 1800, f"Incomplete exhaustive check: {name}")
    invariants = args.out / "offline-invariants.csv"
    run("-m", "scripts.cpu.dynamic.study_offline_invariants", "--out", invariants)
    require(csv_rows(invariants) == csv_rows(ROOT / "docs/evidence/dynamic-cpu-exact-clock.csv"),
            "Exact-clock bound invariants differ from archived results")
    replay_target = args.out / "entry-replay.csv"
    run("-m", "scripts.cpu.dynamic.study_entry_replay", "--synthetic-only", "--out", replay_target)
    for suffix in ("-synthetic.csv", "-compatibility.csv"):
        require(csv_rows(args.out / ("entry-replay" + suffix)) == csv_rows(
            ROOT / "docs/evidence" / ("dynamic-cpu-entry-replay" + suffix)),
            f"Committed replay evidence differs: {suffix}")
    regenerated = args.out / "unseen-arrival.csv"
    run("-m", "scripts.cpu.dynamic.study_unseen_arrival", "--out", regenerated)
    for suffix in (".csv", "-actions.csv", ".witness.json"):
        actual = args.out / ("unseen-arrival" + suffix)
        archived = ROOT / "docs/evidence" / ("dynamic-cpu-unseen-arrival" + suffix)
        if suffix.endswith(".csv"):
            same = csv_rows(actual) == csv_rows(archived)
        else:
            same = json.loads(actual.read_text(encoding="utf-8")) == json.loads(archived.read_text(encoding="utf-8"))
        require(same, f"Unseen-arrival evidence differs: {suffix}")
    metadata = dict(
        python=sys.version.split()[0], frozen_inputs=len(actual_hashes),
        exhaustive_configs_per_mode=300, five_request_configs_per_mode=18,
        transfer_bound_configs=len(transfer),
        exact_clock_configs=240,
        committed_replay_configs=240, committed_replay_compatibility=840,
        unseen_arrival_families=100, archived_deterministic_results_match=True,
        frozen_source_sha256=sha256(ROOT / "docs/evidence/dynamic-cpu-frozen.json"),
    )
    (args.out / "checks.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("CPU delivery checks passed, including frozen inputs and regenerated evidence.")


if __name__ == "__main__":
    main()
