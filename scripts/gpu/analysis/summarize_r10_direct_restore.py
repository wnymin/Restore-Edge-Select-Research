"""Validate complete paired blocks for exact Qwen3.5 direct device restores."""

import argparse
import csv
import io
import json
from math import isfinite
from pathlib import Path

from scripts.gpu.analysis.summarize_r10_async_qualification import digest
from scripts.gpu.analysis.summarize_r10_overlap import describe, validate

ARMS = {"ram-isolated", "ram-overlap", "split-ssd-overlap"}
DEPTHS = (512, 1024, 1536)


def text(path):
    raw = path.read_bytes()
    return raw.decode("utf-16" if raw.startswith(b"\xff\xfe") else "utf-8-sig")


def summarize(raw, baseline, log, repeats):
    groups = {}
    next_value = 1
    for row in csv.DictReader(io.StringIO(text(raw))):
        rep, depth, order = (int(row[n]) for n in ("rep", "depth", "order"))
        arm = row["arm"]
        validate(arm in ARMS and depth in DEPTHS and -1 <= rep < repeats, row)
        validate((int(row["top_matches"]), float(row["max_error"]), int(row["after_consume_bytes_equal"])) == (8, 0.0, 1), row)
        validate(int(row["timeline_value"]) == next_value, row)
        next_value += 1
        t = {n.removesuffix("_ms"): float(v) for n, v in row.items() if n.endswith("_ms")}
        validate(all(isfinite(x) for x in t.values()), row)
        validate(0 <= t["host_ready"] <= t["dma_submit"] <= t["dma_ready"] <= t["bind"] <= t["consume"] < t["consumed"], row)
        ssd = arm == "split-ssd-overlap"
        pending = int(row["pred_fence_pending_at_dma_ready"])
        if arm == "ram-isolated":
            validate(t["pred_submitted"] == t["pred_done_bound"] == pending == -1, row)
        else:
            validate(0 <= t["pred_submitted"] <= t["host_ready"] and t["dma_ready"] <= t["pred_done_bound"] <= t["bind"] and
                     pending in (0, 1), row)
        l = {n: int(row[n]) for n in ("checkpoint_bytes", "attention_encoded_bytes", "ssd_bytes",
                                         "process_read_bytes", "span_count", "dma_bytes")}
        if ssd:
            validate(t["pred_submitted"] <= t["read_done"] <= t["host_ready"] and
                     l["ssd_bytes"] >= l["checkpoint_bytes"] - l["attention_encoded_bytes"] and
                     l["ssd_bytes"] % 4096 == 0 and l["process_read_bytes"] >= l["ssd_bytes"], row)
        else:
            validate(t["read_done"] == -1 and l["ssd_bytes"] == l["process_read_bytes"] == 0, row)
        validate(0 < l["attention_encoded_bytes"] < l["checkpoint_bytes"] and 0 < l["dma_bytes"] < l["checkpoint_bytes"] and
                 l["span_count"] > 0, row)
        group = groups.setdefault((rep, depth), {})
        validate(arm not in group and order == len(group), row)
        group[arm] = t | l | {"pred_pending": pending}
    validate(set(groups) == {(r, d) for r in range(-1, repeats) for d in DEPTHS} and
             all(set(g) == ARMS for g in groups.values()), "incomplete paired matrix")
    records = {}
    for depth in DEPTHS:
        rows = [r for (rep, d), group in groups.items() if d == depth for r in group.values()]
        sizes = {key: rows[0][key] for key in ("checkpoint_bytes", "attention_encoded_bytes", "span_count", "dma_bytes")}
        validate(all(all(r[k] == sizes[k] for k in sizes) for r in rows), f"inconsistent bytes at {depth}")
        records[depth] = sizes | {"arms": {}}
        for arm in sorted(ARMS):
            xs = [groups[r, depth][arm] for r in range(repeats)]
            probes = {"submit_to_signal_ms": describe([x["dma_ready"] - x["dma_submit"] for x in xs]),
                      "eight_tokens_ms": describe([x["consumed"] - x["consume"] for x in xs])}
            if arm != "ram-isolated":
                probes["predecessor_device_fence_pending_when_upload_signaled"] = sum(x["pred_pending"] for x in xs)
                probes["pred_submit_to_observed_fence_upper_bound_ms"] = describe([x["pred_done_bound"] - x["pred_submitted"] for x in xs])
            else:
                probes["arm_start_to_signal_ms"] = describe([x["dma_ready"] for x in xs])
            if arm == "split-ssd-overlap":
                probes.update(read_from_process_bytes=sorted({x["process_read_bytes"] for x in xs}),
                              read_file_bytes=sorted({x["ssd_bytes"] for x in xs}),
                              post_submit_to_read_complete_ms=describe([x["read_done"] - x["pred_submitted"] for x in xs]))
            records[depth]["arms"][arm] = probes
    ref = [r for r in csv.DictReader(io.StringIO(text(baseline))) if int(r["rep"]) >= 0 and r["arm"] != "recompute-256"]
    validate(len(ref) == 12 and all(int(r["top_matches_cut"]) == 8 and float(r["max_error_cut"]) == 0 and
                 int(r["state_bytes_equal"]) == 1 for r in ref), "same-source baseline restore sanity failed")
    gpu = []
    for prefix in text(log).splitlines():
        if prefix.startswith("r10_direct_resources,"):
            _, phase, depth, free, total = prefix.split(",")
            gpu.append([phase, int(depth), int(free), int(total)])
    validate(len(gpu) == 6 and [r[0] for r in gpu] == ["before-model", "contexts-ready"] + ["pinned-and-contexts"] * 3 + ["end"] and
             all(0 < r[2] <= r[3] for r in gpu), "GPU snapshots missing or invalid")
    return dict(repeats=repeats, formal_uploads=repeats * len(ARMS) * len(DEPTHS),
                formal_matching_greedy_positions=repeats * len(ARMS) * len(DEPTHS) * 8,
                max_abs_logit_error=0.0, final_state_byte_equal=True, same_source_baseline_arms=len(ref),
                sha256={k: digest(p) for k, p in {"raw": raw, "baseline": baseline, "log": log}.items()},
                gpu_budget_snapshots=gpu, results=records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("raw", "baseline", "log", "out"):
        p.add_argument(f"--{flag}", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=12)
    a = p.parse_args()
    validate(a.repeats > 0, "positive repeat count required")
    with a.out.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(summarize(a.raw, a.baseline, a.log, a.repeats), out, indent=2, sort_keys=True)
        out.write("\n")


if __name__ == "__main__":
    main()
