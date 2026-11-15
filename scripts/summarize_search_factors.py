"""Matched workload interventions on the zero-overhead next4 advantage."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import mean, median
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", type=Path)
    args = p.parse_args()
    groups = defaultdict(dict)
    seen = set()
    names = ("cohort", "path", "max_prompt", "trace", "window", "placement", "spacing", "budget", "io_scale")
    axes = {"spacing": ("1.0", "0.25"), "budget": ("128", "384"), "io_scale": ("16", "32")}
    for file in args.inputs:
        with file.open(encoding="utf-8-sig", newline="") as source:
            for r in csv.DictReader(source):
                if r["policy"] != "next4":
                    continue
                identity = tuple(r[k] for k in names)
                if identity in seen:
                    p.error(f"Duplicate configuration: {identity}")
                seen.add(identity)
                for axis, levels in axes.items():
                    if r[axis] not in levels:
                        continue
                    key = (axis, *(r[k] for k in ("cohort", "path", "max_prompt", "placement")),
                           r["io_scale"] if axis != "io_scale" else "16_to_32")
                    case = tuple(r[k] for k in names if k != axis)
                    groups[key].setdefault(case, {})[r[axis]] = float(r["mean_gain_ms"])
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("factor", "cohort", "path", "max_prompt", "placement", "io_scale",
                     "pairs", "mean_delta_ms", "median_delta_ms", "higher_gt1", "lower_gt1"))
    if not groups:
        p.error("No matched data")
    for key, cases in sorted(groups.items()):
        low, high = axes[key[0]]
        if any(set(v) != {low, high} for v in cases.values()):
            p.error(f"Missing paired configuration in {key}")
        delta = [v[high] - v[low] for v in cases.values()]
        writer.writerow((*key, len(delta), round(mean(delta), 6), round(median(delta), 6),
                         sum(x > 1 for x in delta), sum(x < -1 for x in delta)))


if __name__ == "__main__":
    main()
