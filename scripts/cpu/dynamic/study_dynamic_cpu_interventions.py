"""Label one boundary intervention at a time on the local-cost trajectory."""

import argparse
from copy import deepcopy
import csv
from fractions import Fraction as F
from itertools import product
from pathlib import Path

from scripts.cpu.dynamic.dynamic_cpu_inputs import FIXTURE, MODEL, ROOT, Experiment, make_experiment, read, verify_traces, window_data
from restore_edge_select.dynamic_cpu import projected_cost, static_cost


def labelled_alternatives(exp, target=None):
    """Both branches have identical history and use local-cost after the intervention."""
    observed = []

    def capture(rows, base, later):
        observed.append((rows, base, later))
        return None

    local = exp.run("local-cost")
    baseline = exp.run("snapshot-screened", screen_mode="classified", screen_selector=capture,
                       score_candidate=projected_cost)
    local_plan = local["boundaries_tokens"]
    ends = local["completion_ms"]
    assert baseline["boundaries_tokens"] == local_plan and baseline["completion_ms"] == ends
    indices = (i for i, choices in enumerate(exp.eligible) if len(choices) > 1)
    for i, (rows, base, later) in zip(indices, observed, strict=True):
        assert base == local_plan[i] and set(rows) == set(exp.eligible[i])
        if target is not None and i != target:
            continue
        for boundary, row in rows.items():
            if boundary == base:
                continue
            changed = exp.run("local-cost", decision_overrides={i: boundary})
            assert changed["boundaries_tokens"][:i] == local_plan[:i]
            assert changed["boundaries_tokens"][i + 1:] == local_plan[i + 1:]
            deltas = [before - after for before, after in zip(ends, changed["completion_ms"], strict=True)]
            here = rows[base]
            yield dict(
                request=exp.requests[i]["id"], choice_ms=float(exp.arrivals[i] + row["arrival_age_ms"]),
                base=base, alternative=boundary, ahead=row["head_distance"], arrived_after=later,
                free_mib=row["free_reserve_bytes"] / 2**20, fits=int(row["prefetch_fits"]),
                extra_mib=row["extra_reserve_bytes"] / 2**20, ready_mib=row["ready_bytes"] / 2**20,
                reserved_mib=row["shared_reservation_bytes"] / 2**20,
                active_mib=row["active_new_kv_bytes"] / 2**20,
                in_flight_ms=float(row["inflight_ms"]), queued_io_ms=float(row["queue_debt_ms"]),
                overlap_slack_ms=float(row["head_slack_ms"] + row["committed_ahead_compute_ms"]),
                serial_difference_ms=float(static_cost(row) - static_cost(here)),
                estimated_target_gain_ms=float(projected_cost(here) - projected_cost(row)),
                earlier_gain_ms=float(sum(deltas[:i])), target_gain_ms=float(deltas[i]),
                later_gain_ms=float(sum(deltas[i + 1:])), total_gain_ms=float(sum(deltas)),
                worst_request_loss_ms=float(max(0, *(-t for t in deltas))),
            )


def synthetic(fixture, model):
    # Each factor has a shared reference. The other inputs and cache objects
    # stay fixed within that comparison, including the copy cost.
    for scale, budget, (factor, value) in product((1, 11, 12, 16), (144, 160), (
        ("reference", 0), ("arrival", 600), ("arrival", 1000), ("share", 1), ("occupancy", 1),
    )):
        trace = deepcopy(fixture)
        if factor == "share":
            trace["requests"][2] = dict(deepcopy(trace["requests"][1]), id="C")
        elif factor == "arrival":
            trace["requests"][2]["arrival_ms"] = value
        elif factor == "occupancy":
            # One B-only KV block is initially on device. B's reservation
            # occupies fewer bytes when C arrives, and C cannot use this KV.
            trace["scheduling"]["initial_device_object_ids"] = ["B.kv1"]
        data = make_experiment(trace, model, trace, "ssd", scale, budget)
        for row in labelled_alternatives(data, target=2):
            yield dict(source="synthetic", window="", factor=factor, value=value, placement="split",
                       spacing="", path="ssd", rate=scale, budget=budget, **row)


def archive(directory, fixture, model):
    # Mechanism windows are outside the holdout. Configurations reproduce
    # the archived transfer model, with one pressure change at a time.
    trace = "qwen_traceA_blksz_16.jsonl"
    layouts = (("ssd", F(1)), ("split", F(1, 4)))
    for window, placement, spacing, data in window_data(directory, trace, (107, 108, 112), fixture, model, layouts):
        for path, scale in (("ssd", 1), ("ssd", 11), ("ssd", 12), ("ssd", 16), ("all", 12)):
            exp = make_experiment(fixture, model, data, path, scale, 128)
            candidates = list(labelled_alternatives(exp))
            filters = {
                "overlap": lambda r: r["fits"] and r["overlap_slack_ms"] > 0,
                "shared": lambda r: r["ready_mib"] or r["reserved_mib"] or r["active_mib"],
                "capacity": lambda r: r["ahead"] and 0 < r["free_mib"] < r["extra_mib"] <= 128,
            }
            for label, predicate in filters.items():
                matching = [r for r in candidates if predicate(r)]
                positive = max((r for r in matching if r["total_gain_ms"] > 0),
                               key=lambda row: row["total_gain_ms"], default=None)
                negative = max((r for r in matching if r["total_gain_ms"] <= 0),
                               key=lambda row: row["estimated_target_gain_ms"], default=None)
                for outcome, row in (("flip", positive), ("no-flip", negative)):
                    if row is None:
                        continue
                    yield dict(source=trace, window=window, factor=label, value=outcome, placement=placement,
                               spacing=str(spacing), path=path, rate=scale, budget=128, **row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=ROOT / "data/raw/bailian")
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fixture, model = read(FIXTURE), read(MODEL)["models"][0]
    records = list(synthetic(fixture, model))
    if not args.synthetic_only:
        verify_traces(args.trace_dir)
        records.extend(archive(args.trace_dir, fixture, model))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, tuple(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
