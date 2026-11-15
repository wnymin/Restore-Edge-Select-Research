"""Summarize repeated batch-recovery contrasts without counting scenarios twice."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import mean
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--cases", type=int, default=96)
    args = p.parse_args()
    groups = defaultdict(dict)
    for input_path in args.inputs:
        with input_path.open(encoding="utf-8", newline="") as source:
            for r in csv.DictReader(source):
                key = tuple(r[k] for k in ("cohort", "io_scale", "rate_mode", "slots", "horizon", "policy"))
                case = tuple(r[k] for k in ("trace", "window", "spacing", "budget_mib"))
                if case in groups[key]:
                    p.error(f"Duplicate row: {key}/{case}")
                groups[key][case] = r
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("cohort", "io_scale", "rate_mode", "slots", "horizon", "policy", "cases",
                     "mean_gain_ms", "p95_gain_ms", "mean_loss_1ms", "p95_loss_1ms", "worst_mean_gain_ms",
                     "base_delta_vs_one_ms", "batch_ms", "snapshots", "reversals", "reversals_gt1",
                     "reversals_gt30", "changed", "rejected"))
    for key, cases in sorted(groups.items()):
        ref = groups.get((*key[:2], "median", "1", "4", "point"))
        if len(cases) != args.cases or not ref or ref.keys() != cases.keys():
            p.error(f"Incomplete paired group: {key}")
        avg = [float(r["mean_gain_ms"]) for r in cases.values()]
        p95 = [float(r["p95_gain_ms"]) for r in cases.values()]
        base = [float(r["base_mean_ms"]) - float(ref[c]["base_mean_ms"]) for c, r in cases.items()]
        batch = [float(r["batch_ms"]) for r in cases.values()]
        totals = ("snapshots", "reversals", "reversals_gt1", "reversals_gt30", "changed", "rejected")
        writer.writerow((*key, len(cases), round(mean(avg), 6), round(mean(p95), 6), sum(x < -1 for x in avg),
                         sum(x < -1 for x in p95), round(min(avg), 6), round(mean(base), 6), round(mean(batch), 6),
                         *(sum(int(r[k]) for r in cases.values()) for k in totals)))


if __name__ == "__main__":
    main()
