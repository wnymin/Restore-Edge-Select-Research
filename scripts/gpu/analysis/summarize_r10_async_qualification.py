"""Validate real-model R10 upload-signal and resident shared-consumption pilots."""
import argparse
import csv
from hashlib import file_digest
import json
from math import isfinite
from pathlib import Path

from scripts.gpu.analysis.summarize_r10_overlap import CONTROL, describe, read_pairs, validate


def digest(path):
    if path is None:
        return None
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def timeline(off, on):
    off_data, off_meta = read_pairs(off)
    on_data, on_meta = read_pairs(on)
    validate(off_meta == on_meta and set(on_data) == set(off_data), "different shapes")
    counts, nv = 0, 0
    for key, group in sorted(on_data.items()):
        ctrl = off_data[key]
        validate(set(group) == set(ctrl) and CONTROL in group, key)
        for arm in sorted(group, key=lambda arm: group[arm]["order"]):
            x, y = group[arm], ctrl[arm]
            validate(y["trace"] == -1 and x["top"] == y["top"] and x["order"] == y["order"], (key, arm))
            if x["t0"] >= 0:
                validate(x["trace"] == 0 and x["timeline_value"] == nv + 1, (key, arm))
                nv = x["timeline_value"]
                counts += key[0] >= 0
    out = dict(sha256={"off": digest(off), "on": digest(on)}, pinned_payload_bytes=on_meta[0],
               successful_formal_waits=counts, time_basis="host QPC; within-process event observations", results={})
    for suf in (128, 512):
        pairs = [(off_data[k], on_data[k]) for k in sorted(on_data) if k[0] >= 0 and k[1] == suf]
        r = out["results"][suf] = dict(paired_groups=len(pairs), arms={})
        off_aa, on_aa = [], []
        for x, y in pairs:
            off_aa.append(abs((x["compute-a"]["c1"] - x["compute-a"]["c0"]) -
                              (x["compute-a-repeat"]["c1"] - x["compute-a-repeat"]["c0"])))
            on_aa.append(abs((y["compute-a"]["c1"] - y["compute-a"]["c0"]) -
                             (y["compute-a-repeat"]["c1"] - y["compute-a-repeat"]["c0"])))
        r.update(compute_aa_abs_ms={"off": describe(off_aa), "on": describe(on_aa)})
        for arm in pairs[0][1]:
            controls, traced = zip(*[(x[arm], y[arm]) for x, y in pairs])
            entry = r["arms"][arm] = {}
            if traced[0]["c0"] >= 0:
                entry["compute_ms_on_minus_off"] = describe([
                    (b["c1"] - b["c0"]) - (a["c1"] - a["c0"]) for a, b in zip(controls, traced)])
            if traced[0]["t0"] >= 0:
                entry.update(independent_wait_ms=describe([b["signal_observed"] - b["signal_submitted"] for b in traced]),
                             ggml_finish_minus_signal_ms=describe([b["t1"] - b["signal_observed"] for b in traced]),
                             ggml_upload_ms_on_minus_off=describe([
                                 (b["t1"] - b["t0"]) - (a["t1"] - a["t0"]) for a, b in zip(controls, traced)]))
                if traced[0]["c0"] >= 0:
                    entry.update(
                        signal_observed_after_compute_submit_before_compute_end=sum(
                            b["cs"] <= b["signal_submitted"] <= b["signal_observed"] < b["c1"] - 1 for b in traced),
                        ggml_return_at_least_1_ms_before_compute_end=sum(b["t1"] < b["c1"] - 1 for b in traced))
    return out


def resident(path):
    groups = {}
    with path.open(encoding="utf-8-sig") as f:
        for a in csv.DictReader(f):
            rep, depth = int(a["rep"]), int(a["depth"])
            validate(rep >= -1 and depth in (512, 1024, 1536) and a["shape"] == "resident-shared", a)
            validate(all(a[k] == "1" for k in ("check_after_donor_release", "check_after_other_writes",
                                              "check_other_after_write")), a)
            validate(int(a["prefix_bytes"]) > 0, a)
            validate(all(0 <= int(a[k]) <= 8 for k in ("greedy_first", "greedy_second")), a)
            validate(all(isfinite(float(a[k])) and float(a[k]) >= 0 for k in (
                "max_error_first", "max_error_second", "copy_metadata_ms")), a)
            validate((rep, depth) not in groups, a)
            groups[rep, depth] = a
    reps = max(rep for rep, _ in groups) + 1
    validate(reps >= 1 and set(groups) == {(r, d) for r in range(-1, reps) for d in (512, 1024, 1536)}, "missing pairs")
    results = {}
    for depth in (512, 1024, 1536):
        rows = [groups[r, depth] for r in range(reps)]
        results[depth] = dict(formal_pairs=reps, bytes=int(rows[0]["prefix_bytes"]),
            metadata_copy_ms=describe([float(r["copy_metadata_ms"]) for r in rows]),
            max_abs_logit_error=max(float(r[k]) for r in rows for k in ("max_error_first", "max_error_second")),
            matching_greedy_positions=sum(int(r[k]) for r in rows for k in ("greedy_first", "greedy_second")),
            total_positions=16 * reps)
    return dict(sha256=digest(path), results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--resident", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = dict(timeline=timeline(args.off, args.on))
    if args.resident:
        results["resident"] = resident(args.resident)
    with args.out.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(results, f, sort_keys=True, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
