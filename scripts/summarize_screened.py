"""Compare selective queue forecasts, decision overhead and full forecasts."""

import argparse
import csv
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import mean
import sys

MODES = ("disagree_0", "disagree_30", "unsettled_0", "unsettled_30")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", type=Path, nargs="+")
    p.add_argument("--cases-from", type=Path, help="keep only case identifiers in an independent CSV")
    args = p.parse_args()
    def identity(r):
        return (r["placement"], r["trace"], int(r["window"]), Fraction(r["spacing"]),
                int(r["budget_mib"]), Fraction(r["io_scale"]))
    selected = None
    if args.cases_from:
        with args.cases_from.open(encoding="utf-8-sig") as source:
            selected = {identity(r) for r in csv.DictReader(source)}
    groups = defaultdict(dict)
    found = set()
    for path in args.inputs:
        with path.open(encoding="utf-8-sig", newline="") as source:
            for r in csv.DictReader(source):
                ident = identity(r)
                if selected is not None and ident not in selected:
                    continue
                found.add(ident)
                key = (r["placement"], r["io_scale"], r["spacing"], r["budget_mib"])
                case = (r["trace"], r["window"])
                if case in groups[key]:
                    p.error(f"Duplicate case {key}, {case}")
                groups[key][case] = r
    if selected is not None and found != selected:
        p.error(f"Missing {len(selected - found)} requested configurations")
    columns = ("mean_gain_ms", "p95_gain_ms", "timed_mean_gain_ms", "timed_p95_gain_ms", "timed_cpu_ms")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("placement", "io_scale", "spacing", "budget_mib", "policy", "cases", *columns,
                     "zero_loss_cases", "timed_loss_cases", "rollouts", "full_rollouts", "choices_vs_live"))
    for key, cases in sorted(groups.items()):
        rows = list(cases.values())
        total = sum(int(r["live_replays"]) for r in rows)
        modes = (*MODES, "robust", "fast") if all("robust_replays" in r for r in rows) else (*MODES, "fast")
        for mode in modes:
            keys = ([f"{mode}_{c}" for c in columns] if mode != "fast" else
                    ["zero_mean_gain_ms", None, "timed_mean_gain_ms", "timed_p95_gain_ms", "timed_fast_cpu_ms"])
            values = [[float(r[c]) for r in rows if c is not None and r.get(c, "") != ""] for c in keys]
            writer.writerow((*key, mode, len(rows), *(round(mean(v), 6) if v else "" for v in values),
                             sum(x < 0 for x in values[0]), sum(x < 0 for x in values[2]) if values[2] else "",
                             "" if mode == "fast" else sum(int(r[f"{mode}_replays"]) for r in rows),
                             total, "" if mode == "fast" else sum(int(r[f"{mode}_choices_vs_live"]) for r in rows)))


if __name__ == "__main__":
    main()
