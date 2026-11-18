"""Compute matched ratios for solo prefill, one copy lane, and two copy lanes."""

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    args = p.parse_args()
    groups = defaultdict(dict)
    for r in json.loads(args.input.read_text(encoding="utf-8"))["trials"]:
        key = (r["bytes"], r["cached"], r["suffix"], r["rep"])
        mode = tuple(r["lanes"])
        if mode in groups[key]:
            p.error(f"Duplicate trial: {key}/{mode}")
        groups[key][mode] = r
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("bytes", "cached", "suffix", "rep", "solo_prompt_ms", "ram_compute_ratio", "ssd_compute_ratio",
                     "joint_compute_ratio", "joint_vs_slower_single", "joint_vs_product",
                     "ram_single_rate_ratio", "ssd_single_rate_ratio", "ram_joint_rate_ratio", "ssd_joint_rate_ratio",
                     "ram_joint_latency_ratio", "ssd_joint_latency_ratio"))
    for key, group in sorted(groups.items()):
        if set(group) != {(), ("ram",), ("ssd",), ("ram", "ssd")}:
            p.error(f"Incomplete paired block: {key}")
        solo = group[()]["prompt_ms"]
        ram, ssd, joint = (group[k]["prompt_ms"] / solo for k in (("ram",), ("ssd",), ("ram", "ssd")))

        def rates(mode, path):
            trial = group[mode]
            samples = trial["samples"][path]
            return (len(samples["during"]) * 350_000_000 / (len(samples["before"]) * trial["host_interior_ns"]),
                    median(samples["during"]) / median(samples["before"]))

        rsolo, ssolo = (rates((path,), path)[0] for path in ("ram", "ssd"))
        rjoint, rl = rates(("ram", "ssd"), "ram")
        sjoint, sl = rates(("ram", "ssd"), "ssd")
        data = (solo, ram, ssd, joint, joint / max(ram, ssd), joint / (ram * ssd),
                rsolo, ssolo, rjoint, sjoint, rl, sl)
        writer.writerow((*key, *(round(v, 6) for v in data)))


if __name__ == "__main__":
    main()
