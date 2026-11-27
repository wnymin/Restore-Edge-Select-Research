"""Audit equal-span arrival pairs from the frozen R07 holdout archive."""

import gzip
import json
from collections import defaultdict
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from statistics import mean

EVIDENCE = Path(__file__).resolve().parents[2] / "docs" / "evidence"


def gap(row):
    value = Fraction(row["main_exact"])
    lower = Fraction(row["oracle"]["lower_exact"])
    upper = Fraction(row["oracle"]["upper_exact"])
    assert lower <= upper <= value
    return value - upper, value - lower


def main():
    provenance = json.loads((EVIDENCE / "r07-extension-provenance.json").read_bytes())["holdout"]
    archive = (EVIDENCE / provenance["archive"]).read_bytes()
    assert sha256(archive).hexdigest() == provenance["archive_sha256"]
    raw = gzip.decompress(archive)
    assert sha256(raw).hexdigest() == provenance["output_sha256"]

    pairs = defaultdict(dict)
    for line in raw.splitlines():
        row = json.loads(line)
        key = dict(row["key"])
        arrival = key.pop("arrival")
        if arrival not in ("knee-even", "knee-burst"):
            continue
        arms = pairs[tuple(sorted(key.items()))]
        assert arrival not in arms
        arms[arrival] = row

    groups = defaultdict(list)
    for key, arms in sorted(pairs.items()):
        even, burst = arms["knee-even"], arms["knee-burst"]
        for field in ("request_ids", "lengths", "arrival_span", "eligible_hash",
                      "payload", "compute_factor", "execution_bytes"):
            assert even["input"][field] == burst["input"][field], (key, field)
        assert even["input"]["arrival_span"] > 0
        assert even["input"]["arrivals_hash"] != burst["input"]["arrivals_hash"]
        even_lo, even_hi = gap(even)
        burst_lo, burst_hi = gap(burst)
        key = dict(key)
        groups[key["model"], key["rate"]].append((burst_lo - even_hi, burst_hi - even_lo))

    published = json.loads((EVIDENCE / "r07-extension-summary.json").read_bytes())
    expected = {(r["model"], r["rate"]): r for r in published["contrasts"]
                if r["kind"] == "burst-minus-even"}
    assert groups.keys() == expected.keys()
    print("model,rate,pairs,positive,negative,exact_zero,unresolved,mean_lower_ms,mean_upper_ms")
    for key, intervals in sorted(groups.items()):
        lo, hi = map(mean, zip(*intervals))
        positive = sum(a > 0 for a, _ in intervals)
        negative = sum(b < 0 for _, b in intervals)
        zero = sum(a == b == 0 for a, b in intervals)
        unresolved = len(intervals) - positive - negative - zero
        assert len(intervals) == expected[key]["n"]
        assert all(abs(float(x) - y) < 1e-9 for x, y in zip((lo, hi), expected[key]["difference"]))
        print(f"{key[0]},{key[1]},{len(intervals)},{positive},{negative},{zero},{unresolved},"
              f"{float(lo):.9f},{float(hi):.9f}")


if __name__ == "__main__":
    main()
