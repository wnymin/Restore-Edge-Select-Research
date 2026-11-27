"""Replay and verify the real Vulkan arena trace, including its physical quotas."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
from statistics import median

PAGE = 65536
ARMS = {"exact-union", "union-minus-page", "inflight-cancel"}
ACTIVE_ARMS = {"exact-union", "exact-union-active-kv"}
DEPS = {0: {0, 1}, 1: {0, 2, 3}}


def need(condition, detail):
    if not condition:
        raise ValueError(detail)


def rounded(n):
    return (n + PAGE - 1) // PAGE * PAGE


def check_arm(arm, rows, sizes):
    expected_uploads = {"exact-union": [0, 1, 2, 3], "exact-union-active-kv": [0, 1, 2, 3],
                        "union-minus-page": [0, 1, 0, 2, 3], "inflight-cancel": [0, 2, 3]}[arm]
    reuse = arm == "exact-union-active-kv"
    expect_requests = [1] if arm == "inflight-cancel" else [0, 1]
    union = sum(map(rounded, sizes.values()))
    capacity = union - (PAGE if arm == "union-minus-page" else 0)
    leases, ready, sent, copies, consumed, disk_reads, markers = set(), set(), [], {}, [], [], []
    flight = None
    peak = prior_us = last_fence = 0
    events = Counter()
    for i, row in enumerate(rows):
        kind, req, obj, size, fence, status = (row[n] for n in ("event", "request", "object", "bytes", "fence", "status"))
        need(row["event_no"] == i and row["wall_us"] >= prior_us and row["capacity_bytes"] == capacity, row)
        prior_us = row["wall_us"]
        events[kind] += 1
        previous = set().union(*(DEPS[r] for r in leases))
        if kind == "device-arena":
            need(i == 0 and size == capacity, row)
        elif kind == "reserve":
            need(req in DEPS and req not in leases, row)
            leases.add(req)
        elif kind == "reject":
            need(req == 1 and leases == {0} and sum(rounded(sizes[k]) for k in DEPS[0] | DEPS[1]) > capacity, row)
        elif kind == "release":
            need(req in leases and (req in copies or (arm == "inflight-cancel" and req == 0 and flight and flight[0] == 0)), row)
            leases.remove(req)
        elif kind == "upload-submit":
            need(flight is None and obj in previous - ready and fence > last_fence and size == sizes[obj], row)
            sent.append(obj)
            last_fence = fence
            flight = (obj, fence)
        elif kind == "upload-ready":
            need(flight == (obj, fence) and obj in previous and size == sizes[obj], row)
            ready.add(obj)
            flight = None
        elif kind == "handoff-at-sample":
            need(flight == (0, fence) and req == 1 and obj == 0 and leases == {0} and status in (0, 1), row)
            markers.append(status)
        elif kind == "early-consume-rejected":
            need(req == 0 and req in leases and not DEPS[req] <= ready and flight and flight[0] == obj == 0, row)
        elif kind in {"missing-kv-rejected", "wrong-checkpoint-rejected"}:
            need(req == 1 and leases == {0, 1} and events["handoff-at-sample"] == 1 and flight and flight[0] == 0, row)
        elif kind == "disk-read":
            need(obj in (1, 3) and size >= sizes[obj] and size % 4096 == 0, row)
            disk_reads.append((obj, size))
        elif kind == "active-kv-reused":
            need(reuse and req == 1 and obj == 0 and size == sizes[0] and consumed == [0] and DEPS[req] <= ready, row)
        elif kind == "device-copy":
            active_bytes = sizes[0] if reuse and req == 1 else 0
            need(req in leases and DEPS[req] <= ready and req not in copies and not flight and events["pred-end"] == 1 and
                 size == sum(sizes[k] for k in DEPS[req]) - active_bytes and status >= 0 and
                 events["active-kv-reused"] == int(bool(active_bytes)), row)
            copies[req] = (size, status)
        elif kind == "consumed":
            need(req in copies and req not in leases and req not in consumed and status == 1 and size == 8, row)
            consumed.append(req)
        elif kind == "head-demand":
            need(arm == "union-minus-page" and req == 1 and consumed == [0] and leases == {1}, row)
        elif kind == "case-end":
            need(i + 1 == len(rows) and not leases and not flight and not ready, row)
        else:
            need(kind in {"pred-submit", "pred-end"}, row)
        active = set().union(*(DEPS[r] for r in leases))
        need(not flight or flight[0] in active, row)
        ready &= active
        charge = sum(rounded(sizes[k]) for k in active)
        refs = sum(obj in DEPS[r] for r in leases) if obj != -1 else 0
        need(row["charged_bytes"] == charge <= capacity and row["refs"] == refs, row)
        peak = max(peak, charge)
    need(list(copies) == consumed == expect_requests and sent == expected_uploads and len(markers) == 1, (arm, sent, copies))
    need([obj for obj, _ in disk_reads] == [k for k in sent if k in (1, 3)], (arm, disk_reads))
    need(all(events[k] == 1 for k in ("device-arena", "pred-submit", "pred-end", "case-end", "early-consume-rejected")), events)
    need(events["reject"] == events["head-demand"] == int(arm == "union-minus-page"), events)
    need(events["missing-kv-rejected"] == events["wrong-checkpoint-rejected"] == int(arm != "union-minus-page"), events)
    need(events["active-kv-reused"] == int(reuse), events)
    return {"actual_device_arena_bytes": capacity, "peak_charged_bytes": peak, "observed_device_pending_on_handoff": markers[0] == 0,
            "upload_bytes": sum(sizes[k] for k in sent), "disk_bytes": sum(n for _, n in disk_reads), "copies": copies,
            "shared_block_uploads": sent.count(0), "exact_logit_positions": len(consumed) * 8}


def summarize(data, repeats, active_kv=False):
    arms = ACTIVE_ARMS if active_kv else ARMS
    raw = data.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-16" if raw.startswith(b"\xff\xfe") else "utf-8-sig")))
    groups, sizes, fences = defaultdict(list), {}, []
    for fields in reader:
        row = {k: v if k in ("arm", "event") else int(v) for k, v in fields.items()}
        groups[row["rep"], row["arm"]].append(row)
        if row["event"] == "upload-submit":
            obj = row["object"]
            need(obj in (0, 1, 2, 3) and sizes.setdefault(obj, row["bytes"]) == row["bytes"], row)
            fences.append(row["fence"])
    need(set(groups) == {(r, a) for r in range(-1, repeats) for a in arms} and set(sizes) == {0, 1, 2, 3}, "missing repeats or objects")
    need(fences == list(range(1, len(fences) + 1)), "native command completion values were lost or repeated")
    cases = [(rep, arm, check_arm(arm, rows, sizes)) for (rep, arm), rows in groups.items() if rep >= -1]
    result = {}
    for arm in sorted(arms):
        xs = [x for rep, name, x in cases if rep >= 0 and name == arm]
        stable = {k: xs[0][k] for k in ("actual_device_arena_bytes", "peak_charged_bytes", "upload_bytes", "disk_bytes", "shared_block_uploads")}
        need(all(all(x[k] == v for k, v in stable.items()) for x in xs), arm)
        result[arm] = stable | {"formal_blocks": len(xs), "pending_before_handoff_count": sum(x["observed_device_pending_on_handoff"] for x in xs),
                                "exact_logit_positions": sum(x["exact_logit_positions"] for x in xs),
                                "device_copy_bytes": {r: xs[0]["copies"][r][0] for r in xs[0]["copies"]},
                                "median_device_copy_wall_us": {r: median(x["copies"][r][1] for x in xs) for r in xs[0]["copies"]}}
    answer = {"raw_sha256": sha256(raw).hexdigest(), "page_bytes": PAGE, "object_bytes": sizes,
              "formal_blocks": repeats * len(arms), "formal_exact_logit_positions": sum(r["exact_logit_positions"] for r in result.values()),
              "arms": result}
    if active_kv:
        by_rep = {(r, a): x for r, a, x in cases if r >= 0}
        only_copy_delta = [by_rep[r, "exact-union"]["copies"][1][1] - by_rep[r, "exact-union-active-kv"]["copies"][1][1]
                           for r in range(repeats)]
        answer["active_kv"] = {"avoided_d2d_bytes_per_successor": sizes[0], "observed_copy_wall_us_paired_deltas": only_copy_delta,
                               "median_copy_wall_us_paired_delta": median(only_copy_delta)}
    return answer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=12)
    p.add_argument("--active-kv", action="store_true")
    a = p.parse_args()
    need(a.repeats > 0, "repeat count must be positive")
    answer = summarize(a.raw, a.repeats, a.active_kv)
    with a.out.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(answer, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
