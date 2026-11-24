"""Measure baseline backlog in the paired R07 arrival replay."""

from bisect import bisect_right
from fractions import Fraction
from hashlib import sha256
import gzip
import json
from statistics import mean

from scripts.cpu.replay.study_r07_extension import CONFIG, ROOT, cases, experiment, read, run


def pressure(exp, baseline):
    # Count earlier requests not yet complete at arrival. Completions at
    # that timestamp have released the slot; earlier equal-time arrivals count.
    arrivals = exp.arrivals
    completions = baseline["completion_ms"]
    assert len(arrivals) == len(completions)
    assert completions == sorted(completions)
    backlog = [i - bisect_right(completions, arrival, 0, i)
               for i, arrival in enumerate(arrivals)]
    return dict(mean_earlier_unfinished=mean(backlog), max_earlier_unfinished=max(backlog),
                mean_queue_wait_ms=float(mean(baseline["queue_wait_ms"])),
                backlog_at_arrival=backlog)


def identity(key):
    return tuple(sorted(key.items()))


def main():
    evidence = ROOT / "docs/evidence"
    meta = read(evidence / "r07-extension-provenance.json")["holdout"]
    archive = (evidence / meta["archive"]).read_bytes()
    assert sha256(archive).hexdigest() == meta["archive_sha256"]
    raw = gzip.decompress(archive)
    assert sha256(raw).hexdigest() == meta["output_sha256"]
    frozen = {}
    for line in raw.splitlines():
        row = json.loads(line)
        if row["key"]["arrival"] in {"knee-even", "knee-burst"}:
            key = identity(row["key"])
            assert key not in frozen
            frozen[key] = row

    for key, data, model, scale, _ in cases(read(CONFIG), "holdout"):
        if key["arrival"] not in {"knee-even", "knee-burst"}:
            continue
        exp = experiment(data, key, model, scale)
        baseline = run(exp, "local-cost")
        result = dict(key=key, baseline_exact=str(Fraction(baseline["response_mean_ms"])),
                      arrivals_hash=sha256("|".join(map(str, exp.arrivals)).encode()).hexdigest(),
                      **pressure(exp, baseline))
        row = frozen.pop(identity(key))
        assert result["arrivals_hash"] == row["input"]["arrivals_hash"], key
        assert exp.lengths == row["input"]["lengths"], key
        assert baseline["request_ids"] == row["input"]["request_ids"], key
        assert Fraction(result["baseline_exact"]) == Fraction(row["baseline_exact"]), key
        main_result = run(exp, "entry-replay")
        assert main_result["boundaries_tokens"] == row["plan"], key
        assert main_result["response_mean_ms"] == Fraction(row["main_exact"]), key
        print(json.dumps(result, separators=(",", ":")))

    assert not frozen, "Some frozen arrival cases were not replayed"


if __name__ == "__main__":
    main()
