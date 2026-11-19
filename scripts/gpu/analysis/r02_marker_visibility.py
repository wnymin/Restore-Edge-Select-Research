"""Audit host visibility of one device completion marker, not kernel activity."""
import argparse
from collections import defaultdict
import csv
import json
from math import isfinite
from pathlib import Path
from random import Random
from statistics import median

SEED = 7943
REPEATS = 16
PERIODS_US = (50, 250)
WORK_MS = (0, 33)


def schedule():
    options = [(w, p) for w in WORK_MS for p in PERIODS_US]
    rng = Random(SEED)
    for rep in range(-1, REPEATS):
        rng.shuffle(options)
        for work, period in options:
            yield rep, work, period


def describe(values):
    return [min(values), median(values), max(values)] if values else None


def audit(rows):
    if [tuple(int(r[k]) for k in ("rep", "work_ms", "period_us")) for r in rows] != list(schedule()):
        raise ValueError("Capture does not match diagnostic schedule")
    groups = defaultdict(list)
    for r in rows:
        first, first_end, done, done_end, origin, pending, count = (
            int(r[k]) for k in ("first_ns", "first_end_ns", "done_ns", "done_end_ns", "origin_end_ns",
                                "last_pending_ns", "queries"))
        marker = float(r["device_end_ms"]) * 1e6
        if not (isfinite(marker) and marker >= 0 and 0 <= origin <= first <= first_end <= done_end
                and first <= done <= done_end and count >= 1
                and ((first == done and first_end == done_end and pending == -1) if count == 1
                     else (first <= pending <= done and first_end <= done))):
            raise ValueError(f"Invalid times in {r}")
        if int(r["rep"]) < 0:
            continue
        groups[r["work_ms"] + ":" + r["period_us"]].append((
            count, (done_end - first_end) / 1e3, (done_end - marker - origin) / 1e3,
            (done_end - marker) / 1e3,
            None if pending < 0 else (pending - marker - origin) / 1e3,
            None if pending < 0 else (done_end - pending) / 1e3))
    return {key: dict(n=len(group), initially_done=sum(x[0] == 1 for x in group),
                      queries=describe([x[0] for x in group]),
                      first_to_done_us=describe([x[1] for x in group]),
                      done_return_minus_marker_lo_us=describe([x[2] for x in group]),
                      done_return_minus_marker_hi_us=describe([x[3] for x in group]),
                      pending_to_done_bracket_width_us=describe([x[5] for x in group if x[5] is not None]),
                      last_pending_start_minus_marker_lo_us=describe([x[4] for x in group if x[4] is not None]))
            for key, group in sorted(groups.items())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture", type=Path)
    args = p.parse_args()
    with args.capture.open(newline="", encoding="utf8") as f:
        print(json.dumps(audit(list(csv.DictReader(f))), indent=2))


if __name__ == "__main__":
    main()
