"""Audit single-action counterfactuals for the frozen B481 tail witness."""

import argparse
from fractions import Fraction
import gzip
import json
from itertools import combinations

from scripts.cpu.replay.study_r08_capacity import CONFIG, ROOT, cases, open_exp, read, run

TARGETS = (15, 16)


def audit(exp, archived, max_edits):
    base = run(exp, "local-cost")
    frozen = run(exp, "entry-replay")
    base_plan, plan = base["boundaries_tokens"], frozen["boundaries_tokens"]
    assert base["response_mean_ms"] == Fraction(archived["base_exact"])
    assert frozen["response_mean_ms"] == Fraction(archived["main_exact"])
    assert plan == archived["plan"]
    assert exp.run("fixed", boundaries=plan)["completion_ms"] == frozen["completion_ms"]

    def metrics(result):
        loss = [end - prior for end, prior in zip(result["completion_ms"], base["completion_ms"], strict=True)]
        return dict(mean_loss_ms=float(result["response_mean_ms"] - base["response_mean_ms"]),
                    worst_loss_ms=float(max(loss)), target_losses_ms=[float(loss[i]) for i in TARGETS],
                    late_indices=[i for i, value in enumerate(loss) if value > 10])

    def phases(result, i):
        queue = result["queue_wait_ms"][i]
        prepare = result["head_to_compute_ms"][i]
        compute = result["completion_ms"][i] - exp.arrivals[i] - queue - prepare
        start = exp.arrivals[i] + queue + prepare
        transfers = [dict(request=t["request"], object=t["object"], path=t["path"], mode=t["mode"],
                          overlap_ms=float(min(t["end_ms"], result["completion_ms"][i]) -
                                           max(t["start_ms"], start)))
                     for t in result["transfers"] if t["start_ms"] < result["completion_ms"][i]
                     and t["end_ms"] > start]
        return dict(queue_ms=float(queue), prepare_ms=float(prepare), compute_ms=float(compute),
                    transfers_during_compute=transfers)

    actions = [(i, c) for i, choices in enumerate(exp.eligible) for c in choices if c != plan[i]]

    def intervene(actions):
        overrides = dict(actions)
        result = run(exp, "entry-replay", overrides=overrides)
        assert all(result["boundaries_tokens"][i] == c for i, c in actions)
        changed = [i for i, value in enumerate(result["boundaries_tokens"]) if value != plan[i]]
        return dict(actions=[dict(index=i, request=exp.requests[i]["id"], original=plan[i], forced=c)
                             for i, c in actions], changed=changed, **metrics(result))

    by_depth = {}
    for depth in range(1, max_edits + 1):
        by_depth[depth] = [intervene(group) for group in combinations(actions, depth)
                           if len({i for i, _ in group}) == depth]
    ordered = sorted([metrics(frozen), *(row for rows in by_depth.values() for row in rows)],
                     key=lambda row: (row["worst_loss_ms"], row["mean_loss_ms"]))
    frontier = []
    for row in ordered:
        if not frontier or row["mean_loss_ms"] < frontier[-1]["mean_loss_ms"]:
            frontier.append(row)
    baseline_edits = {i: a for i, (a, b) in enumerate(zip(base_plan, plan, strict=True)) if a != b}
    baseline_control = run(exp, "entry-replay", overrides=baseline_edits)
    for field in ("boundaries_tokens", "completion_ms", "transfers", "prefetch_reserved_at_ms"):
        assert baseline_control[field] == base[field], field

    return dict(first_divergence=min(baseline_edits), baseline_control_edits=baseline_edits,
                targets=[dict(index=i, request=exp.requests[i]["id"], base_boundary=base_plan[i],
                              frozen_boundary=plan[i], base=phases(base, i), frozen=phases(frozen, i))
                         for i in TARGETS],
                frozen=metrics(frozen), trials_by_depth=by_depth, frontier=frontier)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-edits", type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    cfg = read(CONFIG)
    cfg["holdout"] = {"qwen_traceB_blksz_16.jsonl": [481]}
    match = dict(model="27b", rate=16, section="fixed", level="512")
    path = ROOT / "docs/evidence/r08-holdout.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as source:
        archived, = (r for r in map(json.loads, source) if r["key"]["window"] == 481
                     and r["key"].items() >= match.items())
    key, data, model, scale = next(case for case in cases(cfg, "holdout") if case[0] == archived["key"])
    exp = open_exp(data, model, key["rate"], key["capacity_bytes"], scale)
    print(json.dumps(audit(exp, archived, args.max_edits), indent=2))


if __name__ == "__main__":
    main()
