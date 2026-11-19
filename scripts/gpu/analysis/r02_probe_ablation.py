"""Recompute paired perturbation of the R02 GPU observation probe."""

import argparse
from collections import defaultdict
import csv
import json
from math import isfinite
from pathlib import Path
from random import Random
from statistics import median

SEED = 7942
REPEATS = 24
WINDOW_MS = 33
CELLS = ((80, 7), (80, 30), (112, 7))
ARMS = ("shallow", "deep")
METRICS = ("host_ms", "release_ms", "ready_ms", "target_ms", "peer_c_ms", "peer_d_ms")


def schedule():
    rng = Random(SEED)
    pairs = [(q, w, arm) for q, w in CELLS for arm in ARMS]
    for rep in range(-1, REPEATS):
        rng.shuffle(pairs)
        flips = [rng.randrange(2) for _ in pairs]
        for (q, w, arm), first in zip(pairs, flips):
            for on in (first, 1 - first):
                yield rep, q, w, arm, on


def stats(values):
    return dict(n=len(values), median=median(values), span=[min(values), max(values)])


def audit(rows):
    outcomes = defaultdict(list)
    def order(r):
        return int(r["rep"]), int(r["quota_mib"]), int(r["work_ms"]), r["arm"], int(r["probe"])

    if [order(r) for r in rows] != list(schedule()):
        raise ValueError("Rows do not match frozen paired schedule")
    for first, second in zip(rows[::2], rows[1::2], strict=True):
        rep, q, w, arm, first_on = order(first)
        on, off = (first, second) if first_on else (second, first)
        if on["query_status"] not in ("pending", "done", "unknown") or off["query_status"]:
            raise ValueError("Probe status differs from assignment")
        a, b = ([float(r[k]) for k in METRICS] for r in (on, off))
        if not all(isfinite(x) and x >= 0 for x in a + b):
            raise ValueError("Invalid elapsed time")
        for _, release, ready, target, peer_c, peer_d in (a, b):
            if not release <= target <= peer_c <= peer_d or ready > peer_d:
                raise ValueError("Invalid fixture event ordering")
        if rep < 0:
            continue
        d = dict(zip(METRICS, (i - j for i, j in zip(a, b)), strict=True))
        d["score_ms"] = sum(d[k] for k in METRICS[-3:])
        outcomes[q, w, arm].append((first_on, d))
    return {f"{q}:{w}:{arm}": {k: {label: stats(vals) for label, selected in
                (("all", None), ("on_first", 1), ("off_first", 0))
                if (vals := [d[k] for first, d in group if selected is None or first == selected])}
                for k in (*METRICS, "score_ms")} for (q, w, arm), group in sorted(outcomes.items())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture", type=Path)
    args = p.parse_args()
    with args.capture.open(newline="", encoding="utf8") as f:
        print(json.dumps(audit(list(csv.DictReader(f))), indent=2))


if __name__ == "__main__":
    main()
