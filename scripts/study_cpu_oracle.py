"""Measure the offline optimality gap on a controlled three-request trace."""

import argparse
from copy import deepcopy
import csv
from fractions import Fraction
from itertools import product
import json
from pathlib import Path

from simulate import Experiment, ROOT


POLICIES = ("recompute", "longest", "local-cost", "remaining-wait-fast")


def cases(fixture, model):
    # A starts at 0 and finishes at 30 ms under the fixture's compute model.
    for budget, ssd_scale, c_arrival in product(
        (0, 128, 160, 192), (1, 2, 4, 8), (1, 15, 35)
    ):
        trace = deepcopy(fixture)
        trace["requests"][2]["arrival_ms"] = c_arrival
        ssd = trace["cost"]["ssd_to_device"]
        ssd["bytes_per_ms"] = str(Fraction(ssd["bytes_per_ms"], ssd_scale))
        experiment = Experiment(trace, model, budget)
        optimum = experiment.offline()
        for policy in POLICIES:
            result = experiment.run(policy)
            gap = result["response_sum_ms"] - optimum["response_sum_ms"]
            if gap < 0:
                raise RuntimeError(f"{policy} beats exhaustive search: {budget}/{ssd_scale}/{c_arrival}")
            yield {
                "prefetch_mib": budget,
                "ssd_duration_scale": ssd_scale,
                "c_arrival_ms": c_arrival,
                "policy": policy,
                "boundaries": json.dumps(result["boundaries_tokens"], separators=(",", ":")),
                "completion_ms": json.dumps([float(t) for t in result["completion_ms"]], separators=(",", ":")),
                "mean_response_ms": float(result["response_mean_ms"]),
                "offline_mean_ms": float(optimum["response_mean_ms"]),
                "gap_mean_ms": float(gap / len(experiment.requests)),
                "offline_optima": json.dumps(optimum["offline_optima"], separators=(",", ":")),
                "combinations": optimum["combinations_evaluated"],
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    fixture = json.loads((ROOT / "fixtures/phase1-hand-trace.json").read_text(encoding="utf-8"))
    model = json.loads((ROOT / "configs/qwen3.5-4b.json").read_text(encoding="utf-8"))["models"][0]
    rows = list(cases(fixture, model))
    with args.out.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} policy results from {len(rows) // len(POLICIES)} configurations.")


if __name__ == "__main__":
    main()
