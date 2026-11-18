"""Summarize paired queue-objective results without treating repeats as workloads."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--expected-cases", type=int)
    p.add_argument("--cohort", type=int, nargs="+")
    args = p.parse_args()
    groups = defaultdict(dict)
    with args.input.open(encoding="utf-8", newline="") as source:
        for r in csv.DictReader(source):
            key = (int(r["cohort"]), float(r["io_scale"]))
            if args.cohort and key[0] not in args.cohort:
                continue
            case = tuple(r[x] for x in ("trace", "window", "placement", "budget_mib", "spacing"))
            policies = groups[key].setdefault(case, {})
            policy = r["policy"]
            if policy in policies:
                p.error(f"Duplicate: {key}, {case}, {policy}")
            policies[policy] = r
    if not groups or (args.cohort and {k[0] for k in groups} != set(args.cohort)):
        p.error("Missing requested cohorts")
    metrics = ("mean_gain_ms", "p95_gain_ms", "timed_mean_gain_ms", "timed_p95_gain_ms", "decision_ms")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("cohort", "io_scale", "policy", "cases", *metrics, "zero_losses", "timed_losses_1ms",
                     "timed_p95_losses_1ms", "worst_timed_mean_ms", "changes", "forecast_events"))
    modes = ("single", "single30", "next1", "next4", "all")
    for key, cases in sorted(groups.items()):
        if ((args.expected_cases is not None and len(cases) != args.expected_cases)
                or any(policies.keys() != set(modes) for policies in cases.values())):
            p.error(f"Incomplete group: {key}")
        for policy in modes:
            records = [r[policy] for r in cases.values()]
            values = [[float(r[k]) for r in records if r[k] != ""] for k in metrics]
            writer.writerow((*key, policy, len(records), *(round(mean(v), 5) if v else "" for v in values),
                             sum(x < -1e-6 for x in values[0]), sum(x < -1 for x in values[2]) if values[2] else "",
                             sum(x < -1 for x in values[3]) if values[3] else "",
                             round(min(values[2]), 5) if values[2] else "",
                             sum(int(r["choices_vs_single"]) for r in records),
                             sum(int(r["forecast_events"]) for r in records)))


if __name__ == "__main__":
    main()
