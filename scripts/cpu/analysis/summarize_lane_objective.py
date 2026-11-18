"""Summarize topology effects using only matched arrival and budget settings."""

import argparse
import csv
from itertools import groupby
from pathlib import Path
from statistics import mean
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("shared", type=Path)
    p.add_argument("independent", type=Path)
    p.add_argument("coupled", type=Path)
    p.add_argument("joint", type=Path, nargs="?")
    args = p.parse_args()

    def load(path):
        rows = {}
        with path.open(encoding="utf-8", newline="") as source:
            for r in csv.DictReader(source):
                key = (int(r["cohort"]), int(r["io_scale"]), r["placement"], float(r["spacing"]),
                       r["trace"], int(r["window"]), int(r["budget_mib"]), r["policy"])
                if key in rows:
                    p.error(f"Duplicate case: {key}")
                rows[key] = r
        return rows

    shared, independent, coupled = (load(f) for f in (args.shared, args.independent, args.coupled))
    topologies = [("shared", shared), ("independent", independent), ("coupled", coupled)]
    if args.joint:
        topologies.append(("joint", load(args.joint)))
    for policy in ("single", "next4"):
        cases = [{k for k in source if k[-1] == policy} for _, source in topologies]
        if not cases[0] or any(cases[0] != other for other in cases[1:]):
            p.error("Topology cases must match")
    expected = {(*k[:-1], "next4_robust") for k in shared if k[-1] == "next4"}
    for name, source in topologies[2:]:
        if expected != {k for k in source if k[-1] == "next4_robust"}:
            p.error(f"Missing or extra guarded cases for {name}")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("cohort", "io_scale", "placement", "spacing", "topology", "policy", "cases",
                     "mean_gain_ms", "timed_mean_gain_ms", "timed_p95_gain_ms", "timed_mean_loss_1ms",
                     "timed_p95_loss_1ms", "decision_ms", "lane_overlap_ms", "overlap_cases",
                     "both_lanes_prefill_ms", "both_lanes_prefill_cases",
                     "base_delta_vs_shared_ms", "base_delta_vs_independent_ms", "different_untimed_mean_vs_next4"))
    for prefix, keys in groupby(sorted(shared), key=lambda k: k[:4]):
        keys = list(keys)
        for name, topology in topologies:
            for policy in ("single", "next4", "next4_robust"):
                wanted = [(*k[:-1], policy) for k in keys if k[-1] == "single"]
                if policy == "next4_robust" and name not in {"coupled", "joint"}:
                    continue
                if len(wanted) != 48:
                    p.error(f"Expected 48 pairs per placement/spacing group: {prefix}")
                rows = [topology[k] for k in wanted]
                col = lambda f: [float(r[f]) for r in rows]
                base = col("base_mean_ms")
                other = lambda t: [float(t[(*k[:-1], "single")]["base_mean_ms"]) for k in wanted]
                write = lambda v: round(mean(v), 5)
                timed = col("timed_mean_gain_ms")
                tail = col("timed_p95_gain_ms")
                overlap = col("lane_overlap_ms")
                prefill_overlap = [float(r.get("both_prefill_ms", 0)) for r in rows]
                changes = sum(float(r["mean_gain_ms"]) != float(topology[(*k[:-1], "next4")]["mean_gain_ms"])
                              for k, r in zip(wanted, rows, strict=True)) if policy == "next4_robust" else ""
                writer.writerow((*prefix, name, policy, len(rows), write(col("mean_gain_ms")), write(timed),
                                 write(tail), sum(x < -1 for x in timed), sum(x < -1 for x in tail),
                                 write(col("decision_ms")), write(overlap), sum(v > 0 for v in overlap),
                                 write(prefill_overlap), sum(v > 0 for v in prefill_overlap),
                                 write(x - y for x, y in zip(base, other(shared), strict=True)),
                                 write(x - y for x, y in zip(base, other(independent), strict=True)), changes))


if __name__ == "__main__":
    main()
