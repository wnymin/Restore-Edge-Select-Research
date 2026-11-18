"""Report physical staging tradeoffs and paired ranking differences by block."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--trials", action="store_true")
    args = p.parse_args()
    trials = defaultdict(dict)
    for path in args.inputs:
        for r in json.loads(path.read_text(encoding="utf-8"))["trials"]:
            key = (r["total"], r["ssd_reads"], r["rep"])
            c, concurrent = r["boundary"], r["concurrent"]
            if (concurrent, c) in trials[key]:
                p.error(f"Duplicate trial: {key}")
            expected = 0 if r.get("wait_only") else {0: 0, 512: 2, 1536: 3 + r["ssd_reads"]}[c]
            if len(r["transfer_spans"]) != expected or bool(r["peer"]) != concurrent:
                p.error(f"Incomplete staged recovery: {key}")
            spans = r["transfer_spans"]
            if any(a < 0 or b > r["target"][0] or a >= b for a,b,*_ in spans):
                p.error("Physical copy not contained in pre-submit interval")
            peer = (r["peer"][1] - r["peer"][0]) / 1e6 if concurrent else 0
            before = r["target"][0] / 1e6
            target = r["target"][1] / 1e6
            host_overlap = (sum(max(0, min(r["peer"][1], b) - max(r["peer"][0], a))
                                for a,b,*_ in spans) / 1e6) if concurrent else 0
            trials[key][concurrent, c] = dict(target=target, peer=peer, total=target + peer, dispatch=before,
                                             copy=sum(b-a for a,b,*_ in spans) / 1e6, overlap=host_overlap,
                                             peer_tail=r["peer"][1] / 1e6 if concurrent else 0,
                                             prompt=r["target"][2], peer_prompt=r["peer"][2] if concurrent else 0,
                                             planned_wait=r.get("planned_wait_ns", 0) / 1e6)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    if args.trials:
        cols = ("target", "peer", "total", "dispatch", "copy", "overlap", "peer_tail", "prompt", "peer_prompt",
                "planned_wait")
        writer.writerow(("total_tokens", "ssd_reads", "rep", "concurrent", "boundary",
                         *(f"{k}_ms" for k in cols)))
    else:
        writer.writerow(("total_tokens", "ssd_reads", "rep", "solo_winner", "peer_target_winner",
                         "peer_sum_winner", "solo_long_vs_short_ms", "peer_target_long_vs_short_ms",
                         "peer_sum_long_vs_short_ms", "peer_only_long_vs_short_ms",
                         "solo_short_vs_zero_ms", "peer_sum_short_vs_zero_ms"))
    for key, case in sorted(trials.items()):
        if set(case) != {(v, c) for v in (False, True) for c in (0, 512, 1536)}:
            p.error(f"Missing recovery comparison: {key}")
        if args.trials:
            for (concurrent, c), estimates in sorted(case.items()):
                writer.writerow((*key, int(concurrent), c, *(round(estimates[k], 6) for k in cols)))
        else:
            def winner(concurrent, k):
                return min((0, 512, 1536), key=lambda c: case[concurrent, c][k])
            def diff(concurrent, k, c=512, d=1536):
                return round(case[concurrent, c][k] - case[concurrent, d][k], 6)
            writer.writerow((*key, winner(False, "target"), winner(True, "target"), winner(True, "total"),
                             diff(False, "target"), diff(True, "target"), diff(True, "total"), diff(True, "peer"),
                             diff(False, "target", 0, 512), diff(True, "total", 0, 512)))


if __name__ == "__main__":
    main()
