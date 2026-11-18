"""Summarize a randomized paired GPU lag sweep (resident minus deep; positive favors deep)."""

import argparse
import csv
from pathlib import Path
from statistics import mean
import sys

from scripts.gpu.analysis.project_gpu_boundary import read_blocks
from scripts.gpu.analysis.summarize_gpu_boundary import makespan, target, total


def field(name):
    return lambda row: float(row[name])


METRICS = {
    "target_gain_ms": target,
    "peer_gain_ms": field("peer_ms"),
    "sum_gain_ms": total,
    "finish_gain_ms": makespan,
    "prefill_saved_ms": field("target_prompt_ms"),
    "restore_host_saved_ms": field("restore_host_ms"),
    "restore_server_saved_ms": field("restore_server_ms"),
}


def summarize(path):
    groups = read_blocks(path, allow_completed_peer=True)
    for (n, short, long, peer_total, lag), blocks in sorted(groups.items()):
        if sorted(i for i, _ in blocks) != list(range(len(blocks))):
            raise ValueError(f"{path.name}: expected contiguous replicate numbers")
        if any(len({row["greedy_token"] for row in block.values()}) != 1 for _, block in blocks):
            raise ValueError(f"{path.name}: greedy token diverged within a paired block")
        for q in (0, 1):
            pairs = [(block[q, "resident"], block[q, "fresh"]) for _, block in blocks]
            gains = {k: [metric(a) - metric(b) for a, b in pairs] for k, metric in METRICS.items()}
            active = sum(all(float(row["peer_ms"]) > float(row["actual_peer_lead_ms"])
                             for row in pair) for pair in pairs) if q else 0
            gains["target_interaction_ms"] = [
                gain - (target(block[0, "resident"]) - target(block[0, "fresh"]))
                for gain, (_, block) in zip(gains["target_gain_ms"], blocks, strict=True)
            ]
            yield dict(total_tokens=n, shallow=short, deep=long, peer_total_tokens=peer_total,
                       lag_ms=lag, peer_scheduled=q, pairs=len(pairs), both_peer_active=active,
                       **{k: round(mean(v), 6) for k, v in gains.items()},
                       **{k.replace("_gain_ms", "_deep_wins"): sum(x > 0 for x in v)
                          for k, v in gains.items() if k.endswith("_gain_ms")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        rows = [(path.name, row) for path in args.inputs for row in summarize(path)]
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    fields = ["input", *rows[0][1]]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for name, row in rows:
        writer.writerow(dict(input=name, **row))


if __name__ == "__main__":
    main()
