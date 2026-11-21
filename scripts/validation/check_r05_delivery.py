"""Validate the registered R05 CPU portfolio result."""
import csv
import gzip
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"


def main():
    manifest = json.loads((ROOT / "docs/evidence/r05-manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        data = (ROOT / name).read_bytes()
        assert len(data) == info["bytes"] and hashlib.sha256(data).hexdigest() == info["sha256"], name
    def rows(name):
        path = ROOT / "docs/evidence" / (name + ".gz")
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    result = rows("r05-cpu04-holdout-final.csv")
    assert len(result) == 5184, len(result)
    assert {r["window"] for r in result} == {str(x) for x in range(350, 362)}
    policies = {r["policy"] for r in result}
    assert {"entry", "portfolio-v4", "adaptive", "mixed-reuse-estimate"} <= policies
    means = {p: statistics.mean(float(r["timed_gain_ms"]) for r in result if r["policy"] == p)
             for p in policies}
    assert means["portfolio-v4"] > means["entry"]

    by_window = defaultdict(lambda: defaultdict(list))
    for r in result:
        by_window[r["window"]][r["policy"]].append(float(r["timed_gain_ms"]))
    deltas = [statistics.mean(v["portfolio-v4"]) - statistics.mean(v["entry"])
              for v in by_window.values()]
    assert sum(x > 0 for x in deltas) >= 10, deltas

    requests = rows("r05-cpu04-holdout-requests.csv")
    grouped = defaultdict(dict)
    for r in requests:
        key = tuple(r[k] for k in ("window", "placement", "spacing", "budget", "io_scale", "request"))
        grouped[key][r["policy"]] = r
    compared = [x for x in grouped.values() if "entry" in x and "portfolio-v4" in x]
    assert len(compared) == 81072, len(compared)
    boundary_changes = sum(x["entry"]["zero_boundary"] != x["portfolio-v4"]["zero_boundary"] for x in compared)
    losses = [float(x["portfolio-v4"]["timed_ms"]) - float(x["entry"]["timed_ms"]) for x in compared]
    assert boundary_changes == 0, boundary_changes
    assert sum(x > 10 for x in losses) == 0
    assert max(losses) <= 10
    print("R05 checked: frozen holdout, positive clustered paired gain, equal boundaries, and request guardrails.")


if __name__ == "__main__":
    main()
