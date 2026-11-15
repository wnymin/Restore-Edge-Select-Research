"""Paired CPU and replay differences from full next4 on the same windows."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import median
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", type=Path)
    args = p.parse_args()
    groups = defaultdict(dict)
    dims = ("cohort", "path", "max_prompt", "io_scale")
    for file in args.inputs:
        with file.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                g = tuple(row[k] for k in dims)
                case = tuple(row[k] for k in ("trace", "window", "placement", "budget", "spacing"))
                by_policy = groups[g].setdefault(case, {})
                if row["policy"] in by_policy:
                    p.error(f"Duplicate policy: {g}/{case}")
                by_policy[row["policy"]] = row
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow((*dims, "policy", "cases", "cpu_median_saved_ms", "cpu_q25_saved_ms", "cpu_q75_saved_ms",
                     "cpu_faster_cases", "timed_mean_delta_ms", "timed_p95_delta_ms",
                     "gain_gt1", "loss_gt1"))
    for g, cases in sorted(groups.items()):
        names = tuple(next(iter(cases.values())))
        if "next4" not in names or any(set(c) != set(names) for c in cases.values()):
            p.error(f"Incomplete policy matrix: {g}")
        for mode in (n for n in names if n != "next4"):
            rows = [(c["next4"], c[mode]) for c in cases.values()]
            cpu = sorted(float(a["decision_cpu_ms"]) - float(b["decision_cpu_ms"]) for a,b in rows)
            response = [float(b["timed_gain_ms"]) - float(a["timed_gain_ms"]) for a,b in rows]
            tail = [float(b["timed_p95_ms"]) - float(a["timed_p95_ms"]) for a,b in rows]
            writer.writerow((*g, mode, len(rows), round(median(cpu),6), round(cpu[len(cpu)//4],6),
                             round(cpu[3*len(cpu)//4],6), sum(x > 0 for x in cpu),
                             round(sum(response)/len(rows),6), round(sum(tail)/len(rows),6),
                             sum(x > 1 for x in response), sum(x < -1 for x in response)))


if __name__ == "__main__":
    main()
