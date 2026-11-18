"""Aggregate matched candidate-lifetime and policy evidence across scenarios."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import mean
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", type=Path, nargs="+")
    p.add_argument("--cases", type=int, default=96)
    args = p.parse_args()
    groups = defaultdict(dict)
    for path in args.inputs:
        with path.open(encoding="utf-8", newline="") as source:
            for r in csv.DictReader(source):
                key = tuple(r[k] for k in ("cohort", "io_scale", "release", "host_mib", "policy"))
                case = tuple(r[k] for k in ("trace", "window", "spacing", "budget_mib"))
                if case in groups[key]:
                    p.error(f"Duplicate case: {key}/{case}")
                groups[key][case] = r
    w = csv.writer(sys.stdout, lineterminator="\n")
    means = ("mean_gain_ms", "p95_gain_ms", "timed_gain_ms", "timed_p95_gain_ms", "cpu_ms")
    totals = ("blocked", "blocked_active", "admitted", "same_source", "new_source", "immediate_handoff",
              "host_hits", "host_evictions", "live_changed", "policy_changed")
    w.writerow(("cohort", "io_scale", "release", "host_mib", "policy", "cases", "base_delta_ms", *means, *totals,
                "zero_loss", "timed_loss_1ms", "timed_p95_loss_1ms", "first_change_earlier_hurt",
                "first_change_current_hurt", "first_change_later_hurt"))
    for key, cases in sorted(groups.items()):
        baseline = groups.get((*key[:2], "start", "0", key[-1]))
        if len(cases) != args.cases or (baseline is not None and baseline.keys() != cases.keys()):
            p.error(f"Incomplete paired section: {key}")
        base_diff = [float(r["base_mean_ms"]) - float(baseline[i]["base_mean_ms"])
                     for i, r in cases.items()] if baseline else []
        values = [[float(r[k]) for r in cases.values() if r[k] != ""] for k in means]
        signs = lambda k: sum(float(r[k]) > 1e-6 for r in cases.values() if r[k] != "")
        w.writerow((*key, len(cases), round(mean(base_diff), 6) if base_diff else "",
                    *(round(mean(x), 6) if x else "" for x in values),
                    *(sum(int(r[k]) for r in cases.values()) for k in totals), sum(x < -1e-6 for x in values[0]),
                    sum(x < -1 for x in values[2]) if values[2] else "",
                    sum(x < -1 for x in values[3]) if values[3] else "",
                    *(signs(k) for k in ("earlier_ms", "current_ms", "later_ms"))))


if __name__ == "__main__":
    main()
