"""Independently recompute published Trace B witnesses from frozen archives."""

import gzip
import json
from fractions import Fraction
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parents[2] / "docs" / "evidence"


def read(name):
    with gzip.open(EVIDENCE / name, "rt", encoding="utf-8") as source:
        rows = {}
        for line in source:
            row = json.loads(line)
            key = tuple(sorted(row["key"].items()))
            assert key not in rows
            rows[key] = row
        return rows


def main():
    holdout = read("r08-holdout.jsonl.gz")
    timed = read("r08-timed.jsonl.gz")

    def key(window, model, rate, capacity):
        return tuple(sorted(dict(trace="qwen_traceB_blksz_16.jsonl", window=window,
                                 model=model, rate=rate, section="fixed",
                                 level=str(capacity), capacity_bytes=capacity * 2**20).items()))

    for capacity, cold, reserved, gain in ((176, 0, 0, 0), (180, 14, 13, 0),
                                           (384, 21, 14, 681.075)):
        row = holdout[key(481, "27b", 16, capacity)]
        best = Fraction(row["oracle"]["lower"])
        assert best == Fraction(row["oracle"]["upper_bound"]) == Fraction(row["main_exact"])
        assert row["feasible_by_size"]["cold"] == cold
        assert row["activity"]["entry-replay"]["reserved"] == reserved
        saved = float(Fraction(row["base_exact"]) - best)
        assert abs(saved - gain) <= 0.0005
        print(f"B481 capacity {capacity} MiB: cold={cold}, reserved={reserved}, saving={saved:.3f} ms")

    for case, target, seen, improvement in ((key(481, "4b", 12, 176), 7, 512, 13.251),
                                            (key(482, "27b", 16, 384), 16, 1024, 222.192)):
        row = holdout[case]
        diagnostic = row["diagnostic"]
        assert diagnostic["target"] == target
        assert diagnostic["window4_choice"] == 1024
        assert diagnostic["all_seen_choice"] == seen
        assert diagnostic["joint_choice"] == 512
        assert all(target < i < diagnostic["observed"]["arrived"] for i in diagnostic["joint_indices"])
        forced = diagnostic["closed_loop_forced_mean_ms"]
        assert abs(forced["1024"] - float(Fraction(row["main_exact"]))) < 1e-8
        saved = forced["1024"] - forced["512"]
        assert abs(saved - improvement) <= 0.0005
        if target == 16:
            best = Fraction(row["oracle"]["lower"])
            assert best == Fraction(row["oracle"]["upper_bound"])
            gap = float(Fraction(row["main_exact"]) - best)
            assert abs(gap - 252.670) <= 0.0005
            print(f"B482 offline gap={gap:.3f} ms, remaining after intervention={gap - saved:.3f} ms")
        print(f"B{dict(case)['window']} target {target}: forced saving={saved:.3f} ms")

    rounds = timed[key(481, "27b", 16, 512)]["timed"]
    losses = [r["policies"]["entry-replay"] for r in rounds]
    assert len(losses) == 6 and all(r["late_indices"] == [15, 16] for r in losses)
    maximum = max(r["max_loss"] for r in losses)
    assert abs(maximum - 36.253) <= 0.0005
    print(f"B481 tail: indices 15 and 16 in all six rounds, maximum={maximum:.3f} ms")


if __name__ == "__main__":
    main()
