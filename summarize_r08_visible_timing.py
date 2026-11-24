"""Summarize paired R08 timing by archived offline certificate and source window."""

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import gzip
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]


def identity(key):
    return tuple(sorted(key.items()))


def archived_cases():
    archive = {}
    for name in ("r08-holdout.jsonl.gz", "r08-replacement.jsonl.gz"):
        with gzip.open(ROOT / "docs/evidence" / name, "rt", encoding="utf-8") as stream:
            for row in map(json.loads, stream):
                key = row["key"]
                if key["trace"] == "qwen_coder_blksz_16.jsonl" and key["window"] == 96:
                    continue
                item = identity(key)
                assert item not in archive
                archive[item] = row
    return archive


def certificate(row):
    base = Fraction(row["base_exact"])
    current = Fraction(row["main_exact"])
    lower = Fraction(row["oracle"]["lower"])
    upper = Fraction(row["oracle"]["upper_bound"])
    assert lower <= upper <= min(base, current)
    if base == lower:
        return "no_room"
    if current == lower:
        return "frozen_optimal"
    if current > upper:
        return "frozen_gap_certified"
    return "remaining_gap_uncertain"


def summarize(path):
    archive = archived_cases()
    groups = defaultdict(Counter)
    sources = defaultdict(Counter)
    paired = defaultdict(Counter)
    worst = {}
    for threshold in ("mean_loss", "p95_loss"):
        worst[threshold] = (float("-inf"), None)
    source = gzip.open(path, "rt", encoding="utf-8-sig") if path.suffix == ".gz" else path.open(encoding="utf-8-sig")
    with source as stream:
        for item in map(json.loads, stream):
            key = item["key"]
            archived = archive.pop(identity(key))
            group = certificate(archived)
            timing = item["timing"]
            assert len(timing) == 6
            gain = median(row["gain_over_frozen_ms"] for row in timing)
            p95 = median(row["p95_loss_vs_frozen_ms"] for row in timing)
            rates = [row["retained_saving"] for row in timing]
            retention_defined = all(rate is not None for rate in rates)
            flags = dict(cases=True, zero_gain=item["zero"]["gain_over_frozen_ms"] > 0,
                         mean_gain=gain > 1, mean_loss=gain < -1, p95_loss=p95 > 1,
                         retention_defined=retention_defined,
                         retention_loss=retention_defined and median(rates) < .99,
                         new_persistent=bool(item["added_persistent"]))
            if group in {"no_room", "frozen_optimal"}:
                assert not flags["zero_gain"], key
            groups[group].update(flags)
            for metric, value in (("mean_loss", -gain), ("p95_loss", p95)):
                if value > worst[metric][0]:
                    worst[metric] = value, key
            if item.get("direct_pairs"):
                assert len(item["direct_pairs"]) == 6
                control = [pair["comparison"] for pair in item["direct_pairs"]]
                control_gain = median(row["gain_over_frozen_ms"] for row in control)
                control_p95 = median(row["p95_loss_vs_frozen_ms"] for row in control)
                paired[group].update(dict(
                    cases=True, comparison_mean_loss=control_gain < -1,
                    comparison_p95_loss=control_p95 > 1,
                    candidate_mean_loss=flags["mean_loss"], candidate_p95_loss=flags["p95_loss"],
                    cpu_saved=median(row["cpu_saved_ms"] for row in item["direct_pairs"]) > 1,
                    mean_saved=median(row["mean_saved_ms"] for row in item["direct_pairs"]) > 1,
                    mean_lost=median(row["mean_saved_ms"] for row in item["direct_pairs"]) < -1,
                ))
            sources[key["trace"], key["window"]].update(flags)
    assert not archive
    return dict(by_certificate=groups, paired_by_certificate=paired,
                worst={metric: dict(loss_ms=value, key=key) for metric, (value, key) in worst.items()},
                by_source=[dict(trace=t, window=w, **values) for (t, w), values in sorted(sources.items())])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timing_jsonl", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.timing_jsonl), indent=2))


if __name__ == "__main__":
    main()
