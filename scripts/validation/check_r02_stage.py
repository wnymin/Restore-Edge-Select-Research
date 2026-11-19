"""Recompute the final R02 CPU, GPU, and handoff evidence without CUDA."""

import csv
import json
from pathlib import Path
from statistics import median

from scripts.gpu.analysis import r02_marker_visibility, r02_probe_ablation
from scripts.gpu.experiments import study_r02_gpu_observation as gpu
from scripts.validation import check_r02_cpu

EVIDENCE = Path("docs/evidence")


def read(path):
    return json.loads(path.read_bytes())


def check(condition, label):
    if not condition:
        raise ValueError(f"R02 acceptance evidence differs: {label}")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    check_r02_cpu.main()
    cpu = read(EVIDENCE / "r02-cpu-exchange.json")
    outcomes = {r["id"]: r for r in cpu["outcomes"]}
    ids = [r["id"] for r in sorted(cpu["observations"], key=lambda r: r["capacity_bytes"])]
    check(len(set(ids)) == len(ids) == len(outcomes) == len(cpu["outcomes"])
          and set(ids) == outcomes.keys(), "CPU final case keys")
    check([(outcomes[c]["deep_minus_shallow_sum_ms"], outcomes[c]["best"]) for c in ids]
          == [("9/4", 576), ("-17/4", 1152), ("-17/4", 1152)], "CPU capacity boundary")
    for file in ("r02-cpu-closed-loop.json", "r02-cpu-guarded-timing.json"):
        timing = read(EVIDENCE / file)
        for split in ("design", "holdout"):
            rows = [r for r in timing if r["split"] == split and not r["inject"]]
            check(len(rows) == 216 and all(r["reference_boundaries"] == r["projected_boundaries"]
                  and r["reference_completions"] == r["projected_completions"] for r in rows),
                  f"{file} {split} zero-injection equivalence")
            cost = median(r["projected_decision_ns"] - r["reference_decision_ns"] for r in rows)
            check(cost > 0, f"{file} {split} archived decision cost")
            print(f"CPU {file} {split}: paired decision overhead {cost / 1000:.2f} us")

    cfg = read(Path("configs/r02-gpu-observation-v1.json"))
    fit = gpu.calibration(Path(cfg["calibration"]))
    raw = read(EVIDENCE / "r02-gpu-observation-7941.json")
    original = gpu.audit(cfg, raw, fit)
    recapture = read(EVIDENCE / "r02-gpu-observation-7941-recapture.json")
    check(recapture["fixture_sha256"] == raw["fixture_sha256"], "7941 fixture version")
    check(recapture["outcomes"] != raw["outcomes"], "7941 distinct recapture outcomes")
    for name, audit in (("original", original), ("recapture", gpu.audit(cfg, recapture, fit))):
        check(len(audit["pairs"]) == 48 and len({r["case_id"] for r in audit["pairs"]}) == 48,
              f"7941 {name} paired case keys")
        short = audit["cells"]["tight-short"]
        long = audit["cells"]["tight-long"]
        check(short["observed_delta_ms"]["median"] > 0 > long["observed_delta_ms"]["median"],
              f"7941 {name} window contrast")
        for cell, row in audit["cells"].items():
            tight = cell.startswith("tight-")
            interval = row["methods"]["observable_interval"]
            check(row["n"] == 16 and row["flags"] == {"pending": 32}
                  and row["missing"] == {"active_compute": 32, "active_peer_transfer": 32}
                  and interval["abstain"] == (16 if tight else 0)
                  and interval["right"] == (0 if tight else 16), f"7941 {name} {cell}")
            if not tight:
                check(row["methods"]["isolated"]["right"] == 16, f"7941 {name} wide baseline")
            print(f"GPU 7941 {name} {cell}: median deep minus shallow {row['observed_delta_ms']['median']:.3f} ms")

    probe = r02_probe_ablation.audit(read_csv(EVIDENCE / "r02-gpu-probe-ablation-7942.csv"))
    check(len(probe) == 6 and all(r["score_ms"]["all"]["n"] == 24 for r in probe.values()), "7942 probe pairs")
    markers = r02_marker_visibility.audit(read_csv(EVIDENCE / "r02-gpu-marker-visibility-7943.csv"))
    check(len(markers) == 4 and all(r["n"] == 16 and r["initially_done"] == 0 for r in markers.values()),
          "7943 marker transitions including no-kernel controls")
    print("R02 final evidence verified.")


if __name__ == "__main__":
    main()
