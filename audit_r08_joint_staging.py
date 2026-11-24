"""Audit simultaneous staging as a trigger for public R08 lookahead witnesses."""

from fractions import Fraction
import gzip
import json

from scripts.cpu.replay.study_r07_diagnostics import capture
from scripts.cpu.replay.study_r08_capacity import CONFIG, ROOT, cases, open_exp, read, run
from scripts.validation.audit_r08_visible_follower import diagnostic_executor, audit


OPTIONS = dict(repeats=0, follower_extra=6, sweep=False, ssd_gate=True, sparse_follower=True,
               match_executor=True, require_disagreement=True, pair_triggers=False, reuse_scores=False,
               reuse_prefix=True, capture_inline=True, short_margin_max=105, near_follower=True,
               two_current=True, ssd_compute_min_pct=150)


def analyze(exp, key, row):
    target = 7 if key["window"] == 481 else 16
    snapshot, frozen = capture(exp, target)
    assert frozen["boundaries_tokens"] == row["plan"]
    assert frozen["response_mean_ms"] == Fraction(row["main_exact"])
    result = audit(exp, row, diagnostic_executor(), **OPTIONS)
    event, = (e for e in result["short_gaps"] if e["index"] == target)
    follower = event["follower"]
    deep = max(exp.eligible[follower])
    held = snapshot["state"][8] | set().union(*snapshot["state"][3].values())
    combined = (held | set(exp.candidates[target][event["frozen"]]) |
                set(exp.candidates[follower][deep])) - exp.initial
    updated = run(exp, "entry-replay", overrides={target: event["revised"]})
    assert updated["boundaries_tokens"][target] == event["revised"]
    initial = frozen["prefetch_reserved_at_ms"][follower]
    revised = updated["prefetch_reserved_at_ms"][follower]
    return dict(key=key, target=target, follower=follower, follower_deep=deep,
                combined_bytes=exp.size(combined), pair_fits=exp.size(combined) <= exp.budget,
                frozen=event["frozen"], revised=event["revised"],
                gain_ms=result["zero"]["gain_over_frozen_ms"],
                follower_reserve_original_ms=float(initial) if initial is not None else None,
                follower_reserve_revised_ms=float(revised) if revised is not None else None)


def main():
    selected = {(481, "4b", 12, n) for n in ("128", "176", "180", "256", "384", "512")}
    selected |= {(482, "27b", 16, n) for n in ("384", "512")}

    def match(key):
        return (key["section"] == "fixed" and
                (key["window"], key["model"], key["rate"], key["level"]) in selected)

    with gzip.open(ROOT / "docs/evidence/r08-holdout.jsonl.gz", "rt", encoding="utf-8") as stream:
        archive = {tuple(sorted(row["key"].items())): row for row in map(json.loads, stream) if match(row["key"])}
    cfg = read(CONFIG)
    cfg["holdout"] = {"qwen_traceB_blksz_16.jsonl": [481, 482]}
    for key, data, model, scale in cases(cfg, "holdout"):
        if match(key):
            exp = open_exp(data, model, key["rate"], key["capacity_bytes"], scale)
            print(json.dumps(analyze(exp, key, archive.pop(tuple(sorted(key.items()))))))
    assert not archive


if __name__ == "__main__":
    main()
