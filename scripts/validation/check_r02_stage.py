"""Recompute the final R02 CPU, GPU, and handoff evidence without CUDA."""

import csv
from pathlib import Path
from statistics import median

from scripts.gpu.analysis import r02_marker_visibility, r02_probe_ablation
from scripts.gpu.experiments import study_r02_gpu_observation as gpu
from scripts.validation import check_r02_cpu, export_r02_handoff as handoff


def check(condition, label):
    if not condition:
        raise ValueError(f"R02 acceptance evidence differs: {label}")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    check_r02_cpu.main()
    evidence = handoff.EVIDENCE
    cpu = handoff.read(evidence / "r02-cpu-exchange.json")
    outcomes = {r["id"]: r for r in cpu["outcomes"]}
    ids = [r["id"] for r in sorted(cpu["observations"], key=lambda r: r["capacity_bytes"])]
    check([(outcomes[c]["deep_minus_shallow_sum_ms"], outcomes[c]["best"]) for c in ids]
          == [("9/4", 576), ("-17/4", 1152), ("-17/4", 1152)], "CPU capacity boundary")
    for file in ("r02-cpu-closed-loop.json", "r02-cpu-guarded-timing.json"):
        timing = handoff.read(evidence / file)
        for split in ("design", "holdout"):
            rows = [r for r in timing if r["split"] == split and not r["inject"]]
            check(len(rows) == 216 and all(r["reference_boundaries"] == r["projected_boundaries"]
                  and r["reference_completions"] == r["projected_completions"] for r in rows),
                  f"{file} {split} zero-injection equivalence")
            cost = median(r["projected_decision_ns"] - r["reference_decision_ns"] for r in rows)
            check(cost > 0, f"{file} {split} archived decision cost")
            print(f"CPU {file} {split}: paired decision overhead {cost / 1000:.2f} us")

    cfg = handoff.read(Path("configs/r02-gpu-observation-v1.json"))
    fit = gpu.calibration(Path(cfg["calibration"]))
    raw = handoff.read(handoff.GPU_RAW)
    original = gpu.audit(cfg, raw, fit)
    check(original == handoff.read(handoff.GPU_AUDIT), "7941 archived audit")
    recapture = handoff.read(evidence / "r02-gpu-observation-7941-recapture.json")
    check(recapture["fixture_sha256"] == raw["fixture_sha256"], "7941 fixture version")
    check(recapture["outcomes"] != raw["outcomes"], "7941 distinct recapture outcomes")
    for name, audit in (("original", original), ("recapture", gpu.audit(cfg, recapture, fit))):
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

    probe = r02_probe_ablation.audit(read_csv(evidence / "r02-gpu-probe-ablation-7942.csv"))
    check(len(probe) == 6 and all(r["score_ms"]["all"]["n"] == 24 for r in probe.values()), "7942 probe pairs")
    markers = r02_marker_visibility.audit(read_csv(evidence / "r02-gpu-marker-visibility-7943.csv"))
    check(len(markers) == 4 and all(r["n"] == 16 and r["initially_done"] == 0 for r in markers.values()),
          "7943 marker transitions including no-kernel controls")
    check(handoff.cases() == handoff.read(evidence / "r02-cross-review-v1.json"), "R02-case/v1 examples")
    print("R02 final evidence verified.")


if __name__ == "__main__":
    main()
