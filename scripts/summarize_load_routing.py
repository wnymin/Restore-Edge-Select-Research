"""Aggregate matched window gains and decision-time workload classes."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import mean
import sys


CONDITIONS = ("cohort", "path", "max_prompt", "io_scale")
IDENTITY = ("trace", "window", "placement", "spacing", "budget")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", type=Path, nargs="+")
    p.add_argument("--snapshots", action="store_true")
    p.add_argument("--breakdown", action="store_true", help="separate by placement and arrival spacing")
    p.add_argument("--cases", type=int, help="expected paired configurations per group")
    args = p.parse_args()
    fields = (*CONDITIONS, "placement", "spacing") if args.breakdown else CONDITIONS
    groups = defaultdict(dict)
    for file in args.inputs:
        with file.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                key = tuple(row[k] for k in fields)
                case = tuple(row[k] for k in IDENTITY)
                name = row["label" if args.snapshots else "policy"]
                bucket = groups[key].setdefault(case, {})
                if name in bucket:
                    p.error(f"Duplicate row: {key}/{case}/{name}")
                bucket[name] = row
    if not groups:
        p.error("No evidence rows")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    if args.snapshots:
        writer.writerow((*fields, "label", "decisions", "any_gain", "gain_gt5", "gain_gt5_pct",
                         "sum_predicted_gain_ms", "mean_io_ratio", "mean_ahead", "mean_later", "mean_slack_ms",
                         "mean_debt_ms", "routing_captures_gt5", "depth_captures_gt5"))
    else:
        writer.writerow((*fields, "policy", "cases", "mean_gain_ms", "p95_gain_ms", "timed_gain_ms",
                         "timed_p95_ms", "loss_gt1", "p95_loss_gt1", "worst_gain_ms", "mean_cpu_ms",
                         "baseline_cpu_ms", "forecast_events", "events_pct_next4", "replays", "pruned",
                         "current_calls", "queue_calls"))
    for key, cases in sorted(groups.items()):
        if args.cases is not None and len(cases) != args.cases:
            p.error(f"Missing or extra configurations for {key}: {len(cases)}")
        if args.snapshots:
            if any("all" not in case or sum(int(r["decisions"]) for label, r in case.items() if label != "all")
                   != int(case["all"]["decisions"]) for case in cases.values()):
                p.error(f"Invalid class counts: {key}")
            names = sorted({name for c in cases.values() for name in c})
        else:
            names = tuple(next(iter(cases.values())))
            if "next4" not in names or any(set(case) != set(names) for case in cases.values()):
                p.error(f"Incomplete policy pairs: {key}")
        for name in names:
            rows = [c[name] for c in cases.values() if name in c]
            def total(k):
                return sum(int(r[k]) for r in rows)
            def average(k):
                v = [float(r[k]) for r in rows if r[k] != ""]
                return round(mean(v), 6) if v else ""
            if args.snapshots:
                n = total("decisions")
                gain = total("gains_gt5")
                avg = lambda k: round(sum(float(r[k]) for r in rows) / n, 6)
                writer.writerow((*key, name, n, total("gains"), gain, round(gain / n * 100, 5),
                                 round(sum(float(r["sum_gain"]) for r in rows), 4),
                                 *(avg(k) for k in ("relative", "ahead", "later", "slack_ms", "debt_ms")),
                                 total("routing_captures_gt5"), total("depth_captures_gt5")))
            else:
                timed = bool(rows[0]["timed_gain_ms"])
                used = "timed_gain_ms" if timed else "mean_gain_ms"
                tail = "timed_p95_ms" if timed else "p95_gain_ms"
                gains = [float(r[used]) for r in rows]
                ref = sum(int(c["next4"]["forecast_events"]) for c in cases.values())
                writer.writerow((*key, name, len(rows), *(average(k) for k in
                                 ("mean_gain_ms", "p95_gain_ms", "timed_gain_ms", "timed_p95_ms")),
                                 sum(x < -1 for x in gains), sum(float(r[tail]) < -1 for r in rows),
                                 round(min(gains), 6), average("decision_cpu_ms"), average("baseline_cpu_ms"),
                                 total("forecast_events"), round(100 * total("forecast_events") / ref, 4) if ref else "",
                                 total("replays"), sum(int(r.get("pruned", 0)) for r in rows),
                                 total("current_calls"), total("queue_calls")))


if __name__ == "__main__":
    main()
