"""Summarize physical two-worker transfer trials as within-block ratios."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median
import sys


def p95(v):
    return sorted(v)[(95 * len(v) + 99) // 100 - 1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    args = p.parse_args()
    groups = defaultdict(dict)
    for trial in json.loads(args.input.read_text(encoding="utf-8"))["trials"]:
        case = groups[trial["bytes"], trial["repeat"]]
        lanes = tuple(sorted(trial["samples"]))
        if lanes in case:
            p.error("Duplicate transfer trial")
        case[lanes] = trial
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("bytes", "repeat", "path", "solo_n", "paired_n", "solo_ms", "paired_ms",
                     "median_ratio", "p95_ratio", "rate_ratio", "host_spans_overlap"))
    for (size, repeat), case in sorted(groups.items()):
        if set(case) != {("ram",), ("ssd",), ("ram", "ssd")}:
            p.error(f"Incomplete repeated block: {size}/{repeat}")
        paired = case["ram", "ssd"]
        for path, other in (("ram", "ssd"), ("ssd", "ram")):
            solo = case[path,]
            a, b = ([f - s for s, f, *_ in run["samples"][path]] for run in (solo, paired))
            opposite = paired["samples"][other]
            intersections = sum(any(t < e and f > s for t, f, *_ in opposite)
                                for s, e, *_ in paired["samples"][path])
            writer.writerow((size, repeat, path, len(a), len(b), round(median(a) / 1e6, 6),
                             round(median(b) / 1e6, 6), round(median(b) / median(a), 6),
                             round(p95(b) / p95(a), 6),
                             round((len(b) / paired["window_ns"]) / (len(a) / solo["window_ns"]), 6),
                             intersections))


if __name__ == "__main__":
    main()
