"""Project paired resident-vs-file GPU measurements under versioned production assumptions."""

import argparse
import csv
from collections import defaultdict
from itertools import groupby
import json
from math import isfinite
from pathlib import Path
from statistics import median
import sys


FIELDS = (
    "profile", "input", "prompt_tokens", "short", "long", "peer_total_tokens", "lag_ms", "blocks", "local_prefill_saved_ms",
    "local_response_saved_isolated_ms", "local_response_saved_peer_ms", "local_restore_isolated_ms",
    "local_restore_peer_ms", "scenario", "path", "projected_bytes", "prefill_speedup",
    "projected_prefill_saved_ms", "serial_transfer_ms", "covered_fraction", "extra_ms",
    "projected_net_saved_ms", "projected_choice", "choice_vs_local_isolated", "path_choice_stability",
    "min_net_ms", "max_net_ms",
)


def block_key(row):
    return tuple(int(row[k]) for k in ("total_tokens", "peer_total_tokens", "lag_ms", "rep"))


def read_blocks(path, *, allow_completed_peer=False):
    """Read contiguous paired blocks, normally requiring an active scheduled peer."""
    groups = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for (total, peer_total, lag, rep), block in groupby(csv.DictReader(stream), block_key):
            trials = {}
            for r in block:
                key = (int(r["inflight"]), r["storage"])
                if key in trials:
                    raise ValueError(f"{path.name}: duplicate condition in block {rep}")
                trials[key] = r
                size = int(r["restored_bytes"])
                read = int(r["volume_read_bytes"])
                if key[1] == "fresh":
                    if not (size > 0 and .95 * size <= read <= 1.15 * size):
                        raise ValueError(f"{path.name}: block {rep} has no verified cold restore")
                elif size or not 0 <= read <= 1 << 20:
                    raise ValueError(f"{path.name}: unexpected restore or disk traffic in block {rep}")
                for m in ("target_prompt_ms", "target_ms", "restore_host_ms", "peer_ms"):
                    if not isfinite(float(r[m])) or float(r[m]) < 0:
                        raise ValueError(f"{path.name}: invalid {m} in block {rep}")
                if key[0]:
                    lead = float(r["actual_peer_lead_ms"])
                    if not isfinite(lead) or lead < 0 or (not allow_completed_peer and lead >= float(r["peer_ms"])):
                        raise ValueError(f"{path.name}: invalid active peer in block {rep}")
            if set(trials) != {(q, s) for q in (0, 1) for s in ("none", "resident", "fresh")}:
                raise ValueError(f"{path.name}: expected six resident comparison arms in block {rep}")
            short, long = (int(trials[0, s]["boundary"]) for s in ("resident", "fresh"))
            if not 0 < short < long < total or any(
                    int(trials[q, s]["boundary"]) != b
                    for q in (0, 1) for s, b in (("none", 0), ("resident", short), ("fresh", long))):
                raise ValueError(f"{path.name}: mismatched boundaries in block {rep}")
            group = groups[total, short, long, peer_total, lag]
            if any(i == rep for i, _ in group):
                raise ValueError(f"{path.name}: duplicate block {rep}; pass replicated batches as separate files")
            group.append((rep, trials))
    if not groups:
        raise ValueError(f"{path.name}: no measurements")
    return groups


def validate_profile(profile):
    state = profile["state"]
    if any(type(state[k]) is not int or state[k] < 0 for k in (
            "kv_bytes_per_token", "recurrent_bytes_per_boundary")):
        raise ValueError("State byte counts must be nonnegative integers")
    for key in ("paths", "scenarios"):
        names = [item["name"] for item in profile[key]]
        if not names or len(names) != len(set(names)):
            raise ValueError(f"Expected unique named {key}")
    for path in profile["paths"]:
        if path["source"] not in {"ram", "ssd"} or path["layout"] not in {"whole", "missing"}:
            raise ValueError("Expected RAM or SSD, whole state or missing KV")
    for scenario in profile["scenarios"]:
        if any(not isfinite(scenario[k]) or scenario[k] <= 0 for k in ("prefill_speedup", "h2d_gbps", "ssd_gbps")):
            raise ValueError("Expected positive finite speedup and bandwidth")
        if not (0 <= scenario["covered_fraction"] <= 1 and isfinite(scenario["extra_ms"])
                and scenario["extra_ms"] >= 0):
            raise ValueError("Invalid overlap or extra delay")


def choice(net):
    return "deep" if net > 0 else "resident"


def project(path, min_rep, profile):
    groups = read_blocks(path)
    state = profile["state"]
    for (total, short, long, peer_total, lag), group in sorted(groups.items()):
        blocks = [b for i, b in group if i >= min_rep]
        if not blocks:
            raise ValueError(f"{path.name}: no blocks from {min_rep} for {short}/{long}")

        def saved(q, field):
            return median(float(b[q, "resident"][field]) - float(b[q, "fresh"][field]) for b in blocks)

        local_prefill = saved(0, "target_prompt_ms")
        local_isolated, local_peer = (saved(q, "target_ms") for q in (0, 1))
        local_restore = [median(float(b[q, "fresh"]["restore_host_ms"]) for b in blocks) for q in (0, 1)]
        for route in profile["paths"]:
            tokens = long if route["layout"] == "whole" else long - short
            size = state["recurrent_bytes_per_boundary"] + tokens * state["kv_bytes_per_token"]
            runs = []
            for s in profile["scenarios"]:
                saving = local_prefill / s["prefill_speedup"]
                transfer = size / (s["h2d_gbps"] * 1e6)
                if route["source"] == "ssd":
                    transfer += size / (s["ssd_gbps"] * 1e6)
                net = saving - transfer * (1 - s["covered_fraction"]) - s["extra_ms"]
                runs.append((s, saving, transfer, net))
            low, high = min(r[3] for r in runs), max(r[3] for r in runs)
            stability = "deep_all" if low > 0 else "resident_all" if high <= 0 else "scenario_dependent"
            for s, saving, transfer, net in runs:
                yield (
                    profile["version"], path.name, total, short, long, peer_total, lag, len(blocks),
                    local_prefill, local_isolated, local_peer, *local_restore,
                    s["name"], route["name"], size, s["prefill_speedup"], saving, transfer,
                    s["covered_fraction"], s["extra_ms"], net, choice(net),
                    "same" if choice(net) == choice(local_isolated) else "flip",
                    stability, low, high,
                )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", type=Path, nargs="+")
    p.add_argument("--profile", type=Path, required=True, help="Target assumptions and source calibration for this dataset")
    p.add_argument("--from-rep", type=int, default=0, help="First evaluated paired block")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.from_rep < 0:
        p.error("--from-rep must be nonnegative")
    try:
        profile = json.loads(args.profile.read_text(encoding="utf-8-sig"))
        validate_profile(profile)
        rows = [row for path in args.inputs for row in project(path, args.from_rep, profile)]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.error(str(exc))

    def write(stream):
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(FIELDS)
        writer.writerows(tuple(round(x, 6) if isinstance(x, float) else x for x in row) for row in rows)

    if args.out:
        with args.out.open("w", encoding="utf-8", newline="") as stream:
            write(stream)
    else:
        write(sys.stdout)


if __name__ == "__main__":
    main()
