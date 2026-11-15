"""Export paired first-divergence interventions from a zero-overhead replay."""

import argparse
import csv
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--losses-only", action="store_true")
    args = p.parse_args()
    keys = ("placement", "trace", "window", "spacing", "budget_mib", "io_scale")
    impact = ("earlier_ms", "current_ms", "later_ms", "mean_ms", "later_affected", "later_boundary_changes")
    arms = ("serial_continuation", "approx_continuation", "full_policy_delta")
    fields = ("first_change_index", "request_id", "input_tokens", "serial_boundary", "approx_boundary")
    features = ("head_distance", "head_slack_ms", "committed_ahead_compute_ms",
                "prefetch_fits", "prefetch_reservation_offset_ms", "head_offset_ms",
                "exposed_io_and_copy_ms", "compute_ms")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow((*keys, *fields, *(f"{a}_{v}" for a in arms for v in impact),
                     "approx_advantage_ms", "rollout_advantage_ms",
                     *(f"{c}_{v}" for c in ("serial", "approx") for v in features)))
    seen = set()
    for path in args.inputs:
        with path.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                scenario, first = row["scenario"], row["first_change"]
                case = tuple(scenario[k] for k in keys)
                if case in seen:
                    raise ValueError(f"Duplicate case {case}")
                seen.add(case)
                if not first["attribution"] or (args.losses_only and scenario["approx_vs_serial_mean_ms"] >= 0):
                    continue
                outcome = first["attribution"]
                if abs(outcome["full_policy_delta"]["mean_ms"] + scenario["approx_vs_serial_mean_ms"]) > 1e-7:
                    raise ValueError(f"Mismatched total effect {case}")
                snapshot = first["first_snapshot"]
                values = (*case, *(first[k] for k in fields), *(outcome[a][v] for a in arms for v in impact),
                          snapshot.get("approx_advantage_ms"), snapshot.get("rollout_advantage_ms"),
                          *(snapshot.get(c + "_candidate", {}).get(v)
                            for c in ("serial", "approx") for v in features))
                writer.writerow(round(x, 6) if isinstance(x, float) else x for x in values)


if __name__ == "__main__":
    main()
