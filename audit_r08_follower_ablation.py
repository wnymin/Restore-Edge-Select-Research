"""Ablate follower choices at frozen R08 decision snapshots."""

from fractions import Fraction
import gzip
from itertools import combinations, product
import json
from unittest.mock import patch

from scripts.cpu.replay import study_r07_diagnostics
from scripts.cpu.replay.study_r07_diagnostics import capture, minimizer, trial
from scripts.cpu.replay.study_r08_capacity import CONFIG, ROOT, cases, open_exp, read, run


def residual_snapshot(exp, prior_actions, target, corrected):
    def replay_prior_actions(experiment, policy):
        return run(experiment, policy, overrides=prior_actions)

    with patch.object(study_r07_diagnostics, "run", replay_prior_actions):
        data, actual = capture(exp, target)
    seen = data["state"][2]
    followers = [i for i in range(target + 1, seen) if len(exp.eligible[i]) > 1]
    tests = []
    for varied in [(), *((i,) for i in followers), tuple(followers)]:
        options = [exp.eligible[i] if i in varied else (c,)
                   for i, c in enumerate(data["later"], target + 1)]
        scores = {}
        tails = {}
        for c in exp.eligible[target]:
            scores[c], tails[c] = min((trial(exp, data, (*data["prefix"], c, *tail)), tail)
                                      for tail in product(*options))
        chosen = minimizer(scores, data["fallback"])
        tests.append(dict(varied=varied, choice=chosen,
                          follower_actions={i: tails[chosen][i - target - 1] for i in varied},
                          score_above_best_ms={c: float(v - min(scores.values())) for c, v in scores.items()}))
    assert tests[0]["choice"] == actual["boundaries_tokens"][target]
    assert tests[-1]["choice"] == corrected
    shared = {}
    for test in tests[1:-1]:
        if test["choice"] == corrected:
            follower, = test["varied"]
            chosen = test["follower_actions"][follower]
            shared[follower] = {
                c: sorted(set(exp.candidates[target][c]) & set(exp.candidates[follower][chosen]))
                for c in exp.eligible[target]
            }
    return dict(target=target, arrived=seen, tests=tests, shared_dependencies=shared)


def compare(exp, row):
    diagnostic = row["diagnostic"]
    target = diagnostic["target"]
    followers = diagnostic["joint_indices"]
    data, frozen = capture(exp, target)
    plan = frozen["boundaries_tokens"]
    assert plan == row["plan"]
    original = frozen["response_mean_ms"]
    assert original == Fraction(row["main_exact"])
    base = run(exp, "local-cost")
    assert base["response_mean_ms"] == Fraction(row["base_exact"])
    assert data["state"][2] == diagnostic["observed"]["arrived"]
    assert all(target < i < data["state"][2] for i in followers)

    results = []
    for count in range(len(followers) + 1):
        for varied in combinations(followers, count):
            options = [exp.eligible[i] if i in varied else (c,)
                       for i, c in enumerate(data["later"], target + 1)]
            scores = {}
            tails = {}
            for c in exp.eligible[target]:
                scores[c], tails[c] = min((trial(exp, data, (*data["prefix"], c, *tail)), tail)
                                          for tail in product(*options))
            choice = minimizer(scores, data["fallback"])
            override = {target: choice, **{i: tails[choice][i - target - 1] for i in varied}}

            def replay(actions):
                result = run(exp, "entry-replay", overrides=actions)
                assert all(result["boundaries_tokens"][i] == c for i, c in actions.items())
                losses = [a-b for a,b in zip(result["completion_ms"],base["completion_ms"],strict=True)]
                return dict(gain_ms=float(original - result["response_mean_ms"]),
                            worst_loss_ms=float(max(losses)), late_count=sum(v > 10 for v in losses))

            results.append(dict(varied=varied, choice=choice, overrides=override,
                                score_above_best_ms={c:float(v-min(scores.values())) for c,v in scores.items()},
                                target_only=replay({target: choice}), joint_actions=replay(override)))
    assert results[0]["choice"] == diagnostic["all_seen_choice"]
    assert results[-1]["choice"] == diagnostic["joint_choice"]
    forced = run(exp, "entry-replay", overrides={target: results[-1]["choice"]})
    oracle = row["oracle"]
    residual = None
    if oracle["exact"]:
        optimum = Fraction(oracle["lower"])
        assert exp.run("fixed", boundaries=oracle["plan"])["response_mean_ms"] == optimum
        changes = {i: b for i, (a, b) in enumerate(zip(forced["boundaries_tokens"], oracle["plan"],
                                                       strict=True)) if a != b}
        residual = dict(gap_ms=float(forced["response_mean_ms"] - optimum),
                        remaining_oracle_actions=changes, additional_single_actions={})
        for i, boundary in changes.items():
            result = run(exp, "entry-replay", overrides={target: results[-1]["choice"], i: boundary})
            residual["additional_single_actions"][i] = float(result["response_mean_ms"] - optimum)
        if row["key"]["window"] == 482:
            assert changes == {70: 512}
            residual["snapshot"] = residual_snapshot(exp, {target: results[-1]["choice"]}, 70, changes[70])
    return dict(key=row["key"], target=target, results=results, residual=residual)


def main():
    cfg = read(CONFIG)
    cfg["holdout"] = {"qwen_traceB_blksz_16.jsonl": [481, 482]}
    selected = {(481, "4b", 12, "176"), (482, "27b", 16, "384")}

    def identity(key):
        return key["window"], key["model"], key["rate"], key["level"]

    with gzip.open(ROOT / "docs/evidence/r08-holdout.jsonl.gz", "rt", encoding="utf-8") as source:
        archived = {identity(r["key"]):r for r in map(json.loads, source)
                    if r["key"]["trace"] == "qwen_traceB_blksz_16.jsonl" and r["key"]["section"] == "fixed"
                    and identity(r["key"]) in selected}
    for key, data, model, scale in cases(cfg, "holdout"):
        if key["section"] == "fixed" and identity(key) in selected:
            row = archived.pop(identity(key))
            assert key == row["key"]
            exp = open_exp(data, model, key["rate"], key["capacity_bytes"], scale)
            print(json.dumps(compare(exp, row)))
    assert not archived


if __name__ == "__main__":
    main()
