"""Summarize paired local-cost comparisons with request-level regression counts."""

import argparse
from collections import defaultdict
import csv
from fractions import Fraction
from pathlib import Path
from statistics import mean
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--cases", type=int, default=108)
    args = parser.parse_args()
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("run", "io_scale", "policy", "cases", "positive_mean_cases", "p95_loss_gt10ms_cases",
                     "changed_cases_ssd", "changed_cases_split", "changed_cases_ram", "boundary_changes",
                     "simulated_requests", "requests_slower", "requests_slower_gt10ms", "max_request_loss_ms",
                     "mean_gain_ms", "mean_p95_gain_ms", "timed_mean_gain_ms", "timed_mean_p95_gain_ms",
                     "timed_requests_slower_gt10ms", "timed_max_request_loss_ms"))
    for path in args.inputs:
        groups = defaultdict(dict)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                group = groups[Fraction(row["io_scale"]), row["policy"]]
                key = tuple(row[k] for k in ("trace", "window", "placement", "spacing", "budget"))
                if key in group:
                    parser.error(f"Duplicate case in {path}: {key}")
                group[key] = row
        if not groups or any(len(g) != args.cases or g.keys() != next(iter(groups.values())).keys()
                             for g in groups.values()):
            parser.error(f"Missing or unmatched cases in {path}")
        for (rate, policy), cases in sorted(groups.items()):
            rows = list(cases.values())
            def numbers(name):
                return [float(row[name]) for row in rows if row[name] != ""]

            def count(name):
                return sum(int(row[name]) for row in rows)

            def average(name):
                vals = numbers(name)
                return round(mean(vals), 6) if vals else ""

            def maximum(name):
                vals = numbers(name)
                return round(max(vals), 6) if vals else ""

            timed = numbers("timed_requests_slower_gt10ms")
            writer.writerow((path.stem, str(rate), policy, len(rows),
                             sum(x > 0 for x in numbers("mean_gain_ms")),
                             sum(x < -10 for x in numbers("p95_gain_ms")),
                             *(sum(row["placement"] == tier and int(row["boundary_changes"]) > 0 for row in rows)
                               for tier in ("ssd", "split", "ram")),
                             count("boundary_changes"), count("requests"), count("requests_slower"),
                             count("requests_slower_gt10ms"), maximum("max_request_loss_ms"),
                             average("mean_gain_ms"), average("p95_gain_ms"), average("timed_gain_ms"),
                             average("timed_p95_ms"), int(sum(timed)) if timed else "",
                             maximum("timed_max_request_loss_ms")))


if __name__ == "__main__":
    main()
