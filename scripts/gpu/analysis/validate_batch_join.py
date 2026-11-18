"""Compare simultaneous and delayed-join calibrations on a separate delay."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median
import sys


def pairs(path):
    rows = defaultdict(dict)
    for r in json.loads(path.read_text(encoding="utf-8"))["trials"]:
        shape = tuple(tuple(v) for v in r["shape"])
        key = shape, r["rep"]
        if r["concurrent"] in rows[key]:
            raise ValueError(f"Duplicate pair {key}")
        rows[key][r["concurrent"]] = r["slots"]
    for (shape, rep), modes in rows.items():
        solo, both = modes[False], modes[True]
        start = [v["start_ns"] / 1e6 for v in both]
        end = [v["finish_ns"] / 1e6 for v in both]
        work = [(v["finish_ns"] - v["start_ns"]) / 1e6 for v in solo]
        yield shape, rep, start, end, work


def infer(start, end, work):
    join = max(start)
    first, second = sorted(range(2), key=end.__getitem__)
    overlap = end[first] - join
    if overlap <= 0:
        return None
    before = [join - t for t in start]
    left = [max(0, w - t) for w, t in zip(work, before, strict=True)]
    rates = [0, 0]
    rates[first] = left[first] / overlap
    rates[second] = (left[second] - (end[second] - end[first])) / overlap
    return rates


def predict(start, work, rates):
    join = max(start)
    before = [join - t for t in start]
    if any(w <= t for w, t in zip(work, before, strict=True)):
        return work[:]
    remain = [w - t for w, t in zip(work, before, strict=True)]
    overlap = min(w / r for w, r in zip(remain, rates, strict=True))
    return [t + overlap + max(0, w - overlap * r) for t, w, r in zip(before, remain, rates, strict=True)]


def fit(path, symmetric):
    grouped = defaultdict(list)
    for shape, _, start, end, work in pairs(path):
        rates = infer(start, end, work)
        if rates is None:
            continue
        if symmetric and shape[0] == shape[1]:
            rates = [sum(rates) / 2] * 2
        gap = abs(start[0] - start[1]) / work[0]
        grouped[shape].append((gap, *rates))
    return {shape: tuple(median(a) for a in zip(*rows, strict=True)) for shape, rows in grouped.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("simultaneous", type=Path)
    p.add_argument("delayed", type=Path)
    p.add_argument("holdout", type=Path)
    args = p.parse_args()
    if len({path.resolve() for path in (args.simultaneous, args.delayed, args.holdout)}) != 3:
        p.error("Calibration and holdout paths must differ")
    together = fit(args.simultaneous, True)
    stagger = fit(args.delayed, False)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("slot0_prefix", "slot0_suffix", "slot1_prefix", "slot1_suffix", "rep", "gap_ms",
                     "model", "slot", "solo_ms", "actual_ms", "predicted_ms", "error_ms", "abs_error_ms"))
    for shape, rep, start, end, work in pairs(args.holdout):
        if shape not in together or shape not in stagger or start[1] <= start[0]:
            p.error(f"Unsupported held-out pairing: {shape}/{rep}")
        gap = start[1] - start[0]
        _, *r0 = together[shape]
        limit, *r1 = stagger[shape]
        if limit <= 0:
            p.error(f"No delayed training cases for {shape}")
        levels = {"simultaneous": r0, "hold-delayed": r1, "linear-delay": [
            a + (b - a) * (gap / work[0]) / limit for a, b in zip(r0, r1, strict=True)]}
        for model, rates in levels.items():
            predicted = predict(start, work, [min(1, max(.05, r)) for r in rates])
            for i, guess in enumerate(predicted):
                actual = end[i] - start[i]
                vals = (work[i], actual, guess, guess - actual, abs(guess - actual))
                writer.writerow((*shape[0], *shape[1], rep, round(gap, 6), model, i,
                                 *(round(v, 6) for v in vals)))


if __name__ == "__main__":
    main()
