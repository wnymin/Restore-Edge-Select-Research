"""Fit resident-versus-cold choices on calibration pairs, score held-out GPU measurements."""

import argparse
import csv
from pathlib import Path
from statistics import mean
import sys

from scripts.gpu.analysis.project_gpu_boundary import read_blocks


def target(r):
    return float(r["target_ms"])


def total(r):
    return target(r) + float(r["peer_ms"])


def makespan(r):
    return max(target(r) + float(r["actual_peer_lead_ms"] or 0), float(r["peer_ms"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--calibration", type=int, default=4)
    p.add_argument("--trials", action="store_true")
    args = p.parse_args()
    if args.calibration < 1:
        p.error("Expected a positive calibration count")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    try:
        groups = [(path, read_blocks(path)) for path in args.inputs]
    except (OSError, KeyError, ValueError) as exc:
        p.error(str(exc))
    if args.trials:
        header = None
        for path, _ in groups:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                source = csv.reader(stream)
                columns = next(source)
                if header is None:
                    header = columns
                    writer.writerow(header)
                elif columns != header:
                    p.error(f"Mismatched trial columns: {path}")
                writer.writerows(source)
        return
    writer.writerow(("total_tokens", "short", "long", "peer", "calibration_pairs", "heldout_pairs",
                     "policy", "chosen_boundary", "target_mean_ms", "peer_mean_ms", "sum_mean_ms",
                     "pair_finish_mean_ms", "target_prefill_mean_ms", "pairs_faster_than_isolated",
                     "pairs_slower_than_isolated"))
    for path, trials in groups:
        for (n, short, long, _, _), blocks in trials.items():
            rows = {(i, peer): {int(row["boundary"]): row for (q, _), row in block.items() if q == peer}
                    for i, block in blocks for peer in (0, 1)}
            reps = sorted(i for i, _ in blocks)
            train, test = reps[:args.calibration], reps[args.calibration:]
            if reps != list(range(len(reps))) or not test:
                p.error(f"Missing calibration or held-out pairs: {path}/{short}/{long}")
            choices = (0, short, long)
            isolated = min(choices, key=lambda c: (mean(target(rows[i, 0][c]) for i in train), c))
            candidate = min(choices, key=lambda c: (mean(total(rows[i, 1][c]) for i in train), c))
            # Predeclared agreement gate: one calibration outlier cannot switch the rule.
            active = candidate if sum(total(rows[i, 1][candidate]) < total(rows[i, 1][isolated])
                                      for i in train) * 4 >= 3 * len(train) else isolated
            for q in (0, 1):
                ref = [rows[i,q][isolated] for i in test]
                for name, choice in (("recompute", 0), ("resident", short), ("longest", long),
                                     ("isolated", isolated), ("peer_aware", active if q else isolated)):
                    measured = [rows[i,q][choice] for i in test]
                    metrics = [mean(map(f, measured)) for f in
                               (target, lambda r: float(r["peer_ms"]), total, makespan,
                                lambda r: float(r["target_prompt_ms"]))]
                    delta = [total(a) - total(b) for a,b in zip(ref, measured, strict=True)]
                    writer.writerow((n, short, long, q, len(train), len(test), name, choice,
                                     *(round(x, 6) for x in metrics), sum(x > 0 for x in delta),
                                     sum(x < 0 for x in delta)))


if __name__ == "__main__":
    main()
