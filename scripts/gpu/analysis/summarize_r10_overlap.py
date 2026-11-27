"""Validate paired R10 model/transport measurements and summarize host-timeline evidence."""

import argparse
import csv
from hashlib import file_digest
import io
import json
from math import isfinite
from pathlib import Path
from statistics import median

ARMS = {"compute-a", "compute-a-repeat", "ram-only", "ssd-only", "joint-ram", "joint-ssd"}
CONTROL = "joint-ram-after-submit"
COLUMNS = ("compute_start_ms", "compute_submit_ms", "compute_done_ms", "transfer_start_ms",
           "read_done_ms", "host_ready_ms", "transfer_queued_ms", "transfer_done_ms")


def validate(ok, what):
    if not ok:
        raise ValueError(what)


def read_pairs(path):
    data = path.read_bytes()
    text = data.decode("utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
    pairs, metadata = {}, set()
    for raw in csv.DictReader(io.StringIO(text)):
        rep, suffix, order, size, capacity, read = (int(raw[k]) for k in (
            "rep", "suffix", "order", "payload_bytes", "device_bytes", "read_bytes"))
        arm, direct, top, equal = raw["arm"], int(raw["disk_direct"]), int(raw["top"]), int(raw["device_equal"])
        key = (rep, suffix, arm)
        validate(rep >= -1 and suffix in (128, 512) and arm in ARMS | {CONTROL} and 0 <= order <= len(ARMS), key)
        validate(size > 0 and capacity >= size and direct in (0, 1), key)
        metadata.add((size, capacity, direct))
        ts = [float(raw[k]) for k in COLUMNS]
        validate(all(isfinite(t) for t in ts), key)
        c0, cs, c1, t0, rd, ready, send, t1 = ts
        compute = arm not in {"ram-only", "ssd-only"}
        transfer = not arm.startswith("compute-")
        ssd = arm.endswith("ssd") or arm == "ssd-only"
        validate((0 <= c0 <= cs <= c1 and top >= 0) if compute else (c0 == cs == c1 == top == -1), key)
        validate((0 <= t0 <= ready <= send <= t1 and equal == 1) if transfer else
                 (t0 == rd == ready == send == t1 == equal == -1), key)
        validate((t0 <= rd <= ready and read == (size + 4095) // 4096 * 4096) if ssd else
                 (rd == -1 and read == 0), key)
        validate(arm != CONTROL or cs <= t0, key)
        trace = int(raw.get("trace_status", -1))
        signal_submitted = float(raw.get("signal_submitted_ms", -1))
        signal_observed = float(raw.get("signal_observed_ms", -1))
        timeline_value = int(raw.get("timeline_value", 0))
        validate(trace in (-1, 0, 1, 2, 3) and (transfer or trace == -1), key)
        if trace == 0:
            validate(isfinite(signal_submitted) and isfinite(signal_observed) and
                     send - 1 <= signal_submitted <= signal_observed and timeline_value > 0, key)
        elif trace == -1:
            validate(signal_submitted == signal_observed == -1 and timeline_value == 0, key)
        group = pairs.setdefault((rep, suffix), {})
        validate(arm not in group, key)
        group[arm] = dict(c0=c0, cs=cs, c1=c1, t0=t0, rd=rd, ready=ready, send=send, t1=t1,
                          top=top, order=order, trace=trace, signal_submitted=signal_submitted,
                          signal_observed=signal_observed, timeline_value=timeline_value)
    validate(len(metadata) == 1 and pairs, "unexpected changes in allocation")
    max_rep = max(rep for rep, _ in pairs)
    validate(max_rep >= 0 and set(pairs) == {(rep, suffix) for rep in range(-1, max_rep + 1)
                                            for suffix in (128, 512)}, "missing replicate")
    arms = set(pairs[(-1, 128)])
    validate(arms in (ARMS, ARMS | {CONTROL}), "missing arms")
    for key, group in pairs.items():
        validate(set(group) == arms and {r["order"] for r in group.values()} == set(range(len(arms))), key)
        validate(len({r["top"] for r in group.values() if r["c0"] >= 0}) == 1, key)
    return pairs, metadata.pop()


def describe(values):
    values = sorted(values)
    return dict(median=median(values), min=values[0], max=values[-1],
                p90_nearest_rank=values[(9 * len(values) + 9) // 10 - 1])


def summary(path):
    pairs, (size, capacity, direct) = read_pairs(path)
    with path.open("rb") as f:
        digest = file_digest(f, "sha256").hexdigest()
    result = dict(sha256=digest, serialized_bytes=size, allocated_device_bytes=capacity,
                  unbuffered_read_bytes=(size + 4095) // 4096 * 4096, disk_to_pinned_direct=bool(direct), suffixes={})
    for suffix in (128, 512):
        groups = [g for (rep, n), g in pairs.items() if n == suffix and rep >= 0]
        controls = [[g[k]["c1"] - g[k]["c0"] for k in ("compute-a", "compute-a-repeat")] for g in groups]
        entry = result["suffixes"][str(suffix)] = dict(repeats=len(groups),
            compute_alone_ms=describe([median(x) for x in controls]),
            aa_abs_delta_ms=describe([abs(a - b) for a, b in controls]), paths={})
        kinds = ("ram", "ssd", "ram-after-submit") if CONTROL in groups[0] else ("ram", "ssd")
        for kind in kinds:
            solo = [g[kind.split("-")[0] + "-only"] for g in groups]
            joint = [g["joint-" + kind] for g in groups]
            solo_ms = [t["t1"] - t["t0"] for t in solo]
            metrics = entry["paths"][kind] = dict(
                transfer_alone_ms=describe(solo_ms),
                transfer_joint_ms=describe([t["t1"] - t["t0"] for t in joint]),
                joint_upload_enqueue_call_ms=describe([t["send"] - t["ready"] for t in joint]),
                joint_upload_synchronize_call_ms=describe([t["t1"] - t["send"] for t in joint]),
                slowdown_compute_ms=describe([t["c1"] - t["c0"] - median(c) for t, c in zip(joint, controls)]),
                transfer_done_before_compute_count=sum(t["t1"] < t["c1"] - 1 for t in joint),
                joint_makespan_saving_vs_sum_of_paired_solos_ms=describe([
                    median(c) + s - (max(t["c1"], t["t1"]) - min(t["c0"], t["t0"]))
                    for t, s, c in zip(joint, solo_ms, controls)]),
                joint_upload_finished_before_compute_submit_count=sum(t["t1"] < t["cs"] for t in joint),
            )
            if kind == "ssd":
                metrics["unbuffered_read_alone_ms"] = describe([t["rd"] - t["t0"] for t in solo])
                metrics["unbuffered_read_joint_ms"] = describe([t["rd"] - t["t0"] for t in joint])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with args.out.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(summary(args.csv), f, sort_keys=True, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
