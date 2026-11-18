"""Exact regret lower bounds for identical visible states and different future arrivals."""

import argparse
from copy import deepcopy
from fractions import Fraction as F
from itertools import combinations, product
import json
from pathlib import Path

from scripts.cpu.dynamic.dynamic_cpu_inputs import FIXTURE, MODEL, ROOT, make_experiment, read, sha256
from scripts.cpu.dynamic.study_dynamic_cpu_holdout import write_csv
from scripts.cpu.replay.study_boundary_factors import predict


ARRIVALS = (2, 300, 800, 1200, 1500, 1800)


def randomized_pair(a, b):
    """Minimize the worse of two expected regrets over the action simplex."""
    choices = tuple(a)
    options = [(max(a[c], b[c]), {c: F(1)}) for c in choices]
    for left, right in combinations(choices, 2):
        slope = a[left] - a[right] - b[left] + b[right]
        if not slope:
            continue
        probability = (b[right] - a[right]) / slope
        if 0 <= probability <= 1:
            x = probability * a[left] + (1 - probability) * a[right]
            y = probability * b[left] + (1 - probability) * b[right]
            assert x == y
            options.append((x, {left: probability, right: 1 - probability}))
    value, probabilities = min(options, key=lambda item: item[0])
    assert sum(probabilities.values()) == 1
    assert max(sum(probabilities[c] * row[c] for c in probabilities) for row in (a, b)) == value
    return value, probabilities


def scenario(fixture, model, path, rate, budget, shared, arrival):
    trace = deepcopy(fixture)
    if shared:
        trace["requests"][2] = dict(deepcopy(trace["requests"][1]), id="C")
    trace["requests"][2]["arrival_ms"] = arrival
    exp = make_experiment(trace, model, trace, path, rate, budget)
    snapshots = []

    def observe(rows, base, later):
        if not snapshots:
            assert later == 0
            snapshots.append(rows)
        return None

    exp.run("snapshot-screened", screen_mode="classified", screen_selector=observe,
            score_candidate=lambda row: predict("fifo_snapshot", row))
    scores, plans = {}, {}
    for plan in product(*exp.eligible):
        run = exp.run("fixed", boundaries=plan)
        action = plan[1]
        score = run["response_mean_ms"]
        if action not in scores or score < scores[action]:
            scores[action], plans[action] = score, plan
    return snapshots[0], scores, plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fixture, model = read(FIXTURE), read(MODEL)["models"][0]
    summaries, actions = [], []
    best_witness = None
    for path, rate, budget, shared in product(
            ("ssd", "all"), (1, 11, 12, 16, 32), (0, 70, 128, 160, 192), (False, True)):
        histories, all_scores, all_plans = [], [], []
        key = dict(path=path, rate=rate, budget=budget, shared=int(shared))
        for arrival in ARRIVALS:
            state, scores, plans = scenario(fixture, model, path, rate, budget, shared, arrival)
            histories.append(state)
            all_scores.append(scores)
            all_plans.append(plans)
            optimum = min(scores.values())
            for action, score in scores.items():
                actions.append(dict(**key, c_arrival_ms=arrival, b_boundary=action,
                                    best_with_action_ms=float(score), offline_optimum_ms=float(optimum),
                                    regret_ms=float(score-optimum)))
        assert all(state == histories[0] for state in histories)
        choices = tuple(histories[0])
        regret = [{c: score - min(scores.values()) for c, score in scores.items()} for scores in all_scores]
        worst = {c: max(row[c] for row in regret) for c in choices}
        lower = min(worst.values())
        pair, pair_floor = None, F(-1)
        for a, b in combinations(range(len(ARRIVALS)), 2):
            value = min(max(regret[a][c], regret[b][c]) for c in choices)
            if value > pair_floor:
                pair, pair_floor = (a, b), value
        random_floor, probabilities = randomized_pair(*(regret[i] for i in pair))
        assert 0 <= random_floor <= pair_floor
        summaries.append(dict(
            **key, scenarios=len(ARRIVALS), identical_visible_state=1,
            deterministic_worst_regret_ms=float(lower),
            strongest_pair_regret_ms=float(pair_floor),
            pair_randomized_regret_ms=float(random_floor),
            pair_c_arrivals=json.dumps([ARRIVALS[i] for i in pair]),
            best_fixed_b=json.dumps([c for c in choices if worst[c] == lower]),
        ))
        if best_witness is None or pair_floor > best_witness[0]:
            witness = dict(**key, visible_state=histories[0], pair_regret_lower_ms=pair_floor,
                           pair_regret_exact=str(pair_floor), randomized_regret_lower_ms=random_floor,
                           randomized_regret_exact=str(random_floor),
                           optimal_probabilities={c: str(p) for c, p in probabilities.items()},
                           scenarios=[dict(c_arrival_ms=ARRIVALS[i],
                                           offline_optimum_ms=min(all_scores[i].values()),
                                           action_regrets=regret[i], best_plans=all_plans[i]) for i in pair])
            best_witness = pair_floor, witness
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, summaries)
    write_csv(args.out.with_name(args.out.stem + "-actions.csv"), actions)
    args.out.with_suffix(".witness.json").write_text(
        json.dumps(best_witness[1], indent=2, default=float) + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(dict(
        arrivals=ARRIVALS, families=len(summaries), action_rows=len(actions),
        objective="mean response, zero decision overhead; future continuation is allowed to be clairvoyant",
        sources_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in (
            Path(__file__).resolve(), FIXTURE, MODEL, ROOT / "scripts/cpu/dynamic/dynamic_cpu_inputs.py",
            ROOT / "src/restore_edge_select/simulator.py",
            ROOT / "docs/evidence/prefill-vulkan-4k.json",
            ROOT / "docs/evidence/load-staged-ssd-profile.json", ROOT / "configs/overlap-observed-upper.json")},
    ), indent=2) + "\n", encoding="utf-8")
    print(f"{len(summaries)} fixed-cost families; positive minimax lower bounds:"
          f" {sum(r['deterministic_worst_regret_ms'] > 0 for r in summaries)}")
    print(f"Largest two-arrival regret bound: {float(best_witness[0]):.6f} ms")


if __name__ == "__main__":
    main()
