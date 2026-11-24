"""Compare fixed SSD dispatch delays across published R08 source windows."""

from fractions import Fraction
import gzip
import json

from scripts.validation.audit_r08_ssd_dispatch import diagnostic_executor
from scripts.cpu.replay.study_r08_capacity import CONFIG, ROOT, cases, open_exp, read, run


def measure(base, result):
    losses = [done - prior for done, prior in zip(result["completion_ms"], base["completion_ms"], strict=True)]
    return dict(mean_gain_ms=float(base["response_mean_ms"] - result["response_mean_ms"]),
                worst_loss_ms=float(max(losses)), late_count=sum(loss > 10 for loss in losses))


def compare(exp, row, executor):
    base = run(exp, "local-cost")
    frozen = run(exp, "entry-replay")
    plan = frozen["boundaries_tokens"]
    assert plan == row["plan"]
    assert base["response_mean_ms"] == Fraction(row["base_exact"])
    assert frozen["response_mean_ms"] == Fraction(row["main_exact"])

    def execute(gate):
        result = executor(exp, "fixed", 0, plan, 10000, len(plan), ssd_gate=gate)
        assert result["boundaries_tokens"] == plan
        assert result["peak_prefetch_bytes"] <= exp.budget
        return result

    control = execute({})
    for field in ("completion_ms", "transfers", "prefetch_reserved_at_ms"):
        assert frozen[field] == control[field], field

    # Apply the same rule to every future SSD read. At dispatch time the
    # immediately preceding request has already selected its own boundary.
    absolute = {i: (i - 1, 960) for i in range(1, len(plan))}
    relative = {i: (i - 1, exp.recompute(i - 1, plan[i - 1]) * Fraction(4, 5))
                for i in range(1, len(plan))}
    differences = {i for i, (original, chosen) in enumerate(zip(base["boundaries_tokens"], plan, strict=True))
                   if original != chosen}
    selective = {i: gate for i, gate in relative.items() if i in differences}
    return dict(frozen=measure(base, frozen), fixed_960=measure(base, execute(absolute)),
                relative_80pct=measure(base, execute(relative)),
                divergence_only_80pct=measure(base, execute(selective)))


def main():
    paths = [ROOT / "docs/evidence" / name for name in
             ("r08-holdout.jsonl.gz", "r08-replacement.jsonl.gz")]
    match = dict(model="27b", rate=16, section="fixed", level="512")
    cfg = read(CONFIG)
    # The corrected holdout uses Coder 101 in place of 96.
    cfg["holdout"]["qwen_coder_blksz_16.jsonl"] = [97, 98, 99, 101]
    windows = {(trace, window) for trace, values in cfg["holdout"].items() for window in values}
    frozen = {}
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for row in map(json.loads, source):
                key = row["key"]
                window = key["trace"], key["window"]
                if window in windows and key.items() >= match.items():
                    assert window not in frozen
                    frozen[window] = row
    assert frozen.keys() == windows
    executor = diagnostic_executor()
    for key, data, model, scale in cases(cfg, "holdout"):
        if not match.items() <= key.items():
            continue
        row = frozen.pop((key["trace"], key["window"]))
        assert key == row["key"]
        exp = open_exp(data, model, key["rate"], key["capacity_bytes"], scale)
        print(json.dumps(dict(key=key, **compare(exp, row, executor))))
    assert not frozen


if __name__ == "__main__":
    main()
