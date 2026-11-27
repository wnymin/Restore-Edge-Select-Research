"""Trace frozen B481 recovery reservations through actual executor events."""

from collections import Counter
from fractions import Fraction
import gzip
import json

from scripts.cpu.replay.study_r08_capacity import CONFIG, ROOT, cases, open_exp, read, run


def audit(exp, archived):
    online = run(exp, "entry-replay")
    plan = online["boundaries_tokens"]
    assert plan == archived["plan"]
    assert online["response_mean_ms"] == Fraction(archived["main_exact"])
    fixed = exp.run("fixed", boundaries=plan)
    for field in ("boundaries_tokens", "completion_ms", "prefetch_reserved_at_ms",
                  "queue_wait_ms", "head_to_compute_ms", "transfers"):
        assert online[field] == fixed[field], field

    issued = Counter()
    handoffs = {}
    for item in fixed["transfers"]:
        request = item["request"]
        if item["mode"] == "prefetch":
            issued[request] += 1
        elif item["mode"] == "handoff":
            assert request not in handoffs
            handoffs[request] = item

    results = []
    for index, reservation in enumerate(fixed["prefetch_reserved_at_ms"]):
        if reservation is None:
            continue
        request = exp.requests[index]["id"]
        head_time = exp.arrivals[index] + fixed["queue_wait_ms"][index]
        compute_start = head_time + fixed["head_to_compute_ms"][index]
        # At the recovery gate all dependencies are ready. A state copy
        # may then run before suffix compute, so use its start if present.
        recovery_start = handoffs[request]["start_ms"] if request in handoffs else compute_start
        assert reservation <= head_time <= recovery_start <= compute_start
        assert recovery_start <= fixed["completion_ms"][index]
        assert plan[index] > 0
        results.append(dict(request=request, reserve_ms=str(reservation), head_ms=str(head_time),
                            recovery_start_ms=str(recovery_start), prefetch_issues=issued[request],
                            ready_at_head=recovery_start == head_time))
    assert len(results) == archived["activity"]["entry-replay"]["reserved"]
    return results


def main():
    path = ROOT / "docs/evidence/r08-holdout.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as source:
        frozen = {tuple(sorted(row["key"].items())): row for row in map(json.loads, source)}
    cfg = read(CONFIG)
    cfg["holdout"] = {"qwen_traceB_blksz_16.jsonl": [481]}
    target = {"176", "180", "384"}
    for key, data, model, scale in cases(cfg, "holdout"):
        if key["model"] != "27b" or key["rate"] != 16 or key["section"] != "fixed":
            continue
        if key["level"] not in target:
            continue
        target.remove(key["level"])
        exp = open_exp(data, model, key["rate"], key["capacity_bytes"], scale)
        row = frozen[tuple(sorted(key.items()))]
        requests = audit(exp, row)
        print(json.dumps(dict(key=key, requests=requests,
                              cold_fitting=row["feasible_by_size"]["cold"], reserved=len(requests),
                              issued=sum(r["prefetch_issues"] > 0 for r in requests),
                              ready_at_head=sum(r["ready_at_head"] for r in requests),
                              consumed=len(requests))))
    assert not target


if __name__ == "__main__":
    main()
