"""Recompute and check R10 local preflight evidence from raw files."""
import argparse
from collections import defaultdict
import csv
from hashlib import file_digest
import json
from pathlib import Path
from statistics import median

DEPTHS = (512, 1024, 1536)
ARMS = ("recompute-256", "ram-direct", "ram-handoff", "file-handoff", "split-host-handoff")


def sha(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def read_text(path):
    content = path.read_bytes()
    encoding = "utf-16" if content.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    return content.decode(encoding)


def rows(path):
    return list(csv.DictReader(read_text(path).splitlines()))


def restore(path):
    selected = defaultdict(dict)
    for row in rows(path):
        rep, depth, arm = int(row["rep"]), int(row["depth"]), row["arm"]
        if rep >= 0:
            if arm in selected[rep, depth]:
                raise ValueError("duplicate restore arm")
            selected[rep, depth][arm] = row
    if set(selected) != {(rep, depth) for rep in range(4) for depth in DEPTHS}:
        raise ValueError("incomplete restore blocks")
    controls, data = [], defaultdict(list)
    for (rep, depth), arms in selected.items():
        if set(arms) != set(ARMS):
            raise ValueError("incomplete restore arms")
        control = arms["recompute-256"]
        controls.append(float(control["max_error_full"]))
        for arm in ARMS[1:]:
            row = arms[arm]
            if (row["state_bytes_equal"] != "1" or int(row["top_matches_cut"]) != 8 or
                    float(row["max_error_cut"]) != 0 or row["max_error_full"] != control["max_error_full"] or
                    row["top_matches_full"] != control["top_matches_full"]):
                raise ValueError(f"restore diverges from matched prefill control: {rep}/{depth}/{arm}")
            data[depth, arm].append(float(row["load_ms"]))
    return dict(formal_arms=sum(map(len, data.values())), positions_per_arm=8,
                identical_state_arms=sum(map(len, data.values())),
                max_error_from_matched_recompute=0.0, max_error_from_512_batch_recompute=max(controls),
                load_ms_median=[dict(depth=d, arm=a, ms=median(v)) for (d, a), v in sorted(data.items())],
                state_bytes=[dict(depth=d, full_bytes=int(selected[0,d]["recompute-256"]["full_bytes"]),
                                  recurrent_payload_and_metadata_bytes=int(selected[0,d]["recompute-256"]["partial_bytes"]) - 8)
                             for d in DEPTHS])


def isolated(path):
    raw = json.loads(path.read_bytes())
    samples = defaultdict(list)
    for row in raw["samples"]:
        samples[row["kind"], row["bytes"]].append(row["ns"] / 1e6)
    if len(samples) != 18 or {len(times) for times in samples.values()} != {12}:
        raise ValueError("missing isolated path samples")
    return dict(device=raw["device"], platform=raw["platform"], model_sha256=raw["data_sha256"],
                medians=[dict(kind=k, bytes=b, ms=median(v)) for (k, b), v in sorted(samples.items())])


def prefill(path):
    raw = json.loads(read_text(path))
    expected = {(d, p) for d in (0, 512, 1024, 1536) for p in (128, 512, 1024)}
    if len(raw) != len(expected) or {(x["n_depth"], x["n_prompt"]) for x in raw} != expected:
        raise ValueError("missing prefill shapes")
    if any(x["devices"] != "Vulkan1" or len(x["samples_ns"]) != 5 or x["n_gen"] for x in raw):
        raise ValueError("mismatched prefill device or repetitions")
    return [dict(depth=x["n_depth"], suffix=x["n_prompt"], ms=median(x["samples_ns"]) / 1e6) for x in raw]


def staging(path):
    controls, pairs = [], defaultdict(dict)
    for row in rows(path):
        rep, depth = int(row["rep"]), int(row["depth"])
        key = (rep, row["topology"], int(row["quota_mib"]))
        if rep < 0:
            controls.append(row["state_bytes_equal"])
            continue
        if depth in pairs[key]:
            raise ValueError("duplicate staging arm")
        pairs[key][depth] = row
    if set(controls) != {"1"} or len(controls) != 6:
        raise ValueError("missing staging state checks")
    expected = {(i, topology, quota) for i in range(8) for topology, quota in (("split",174), ("shared",174), ("shared",210))}
    if set(pairs) != expected or any(set(pair) != {1152,1280} for pair in pairs.values()):
        raise ValueError("incomplete staging pair matrix")
    groups = defaultdict(list)
    for (rep, topology, quota), pair in sorted(pairs.items()):
        shallow, deep = pair[1152], pair[1280]
        for row in pair.values():
            peak = (int(row["peak_bytes"]) if topology == "shared" else
                    max(int(row["target_object_bytes"]), int(row["peer_object_bytes"])))
            if row["top_token_match"] != "1" or peak > quota * 2**20:
                raise ValueError(f"staging correctness or quota failure: {rep}/{topology}/{quota}")
        if shallow["waited"] != "0" or int(deep["waited"]) != int(topology == "shared" and quota == 174):
            raise ValueError("staging wait disagrees with designed capacity control")
        groups[topology, quota].append(float(deep["target_ms"]) - float(shallow["target_ms"]))
    return [dict(topology=t, quota_mib=q, pairs=len(v), deep_wins=sum(x < 0 for x in v),
                 deep_minus_shallow_ms=median(v), paired_differences_ms=v) for (t,q),v in sorted(groups.items())]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    files = {"restore-split-v2.csv": restore, "isolated-paths.json": isolated,
             "prefill-initial.json": prefill, "stage-native.csv": staging}
    result = {name: dict(sha256=sha(args.directory / name), result=fn(args.directory / name)) for name, fn in files.items()}
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
        out.write("\n")


if __name__ == "__main__":
    main()
