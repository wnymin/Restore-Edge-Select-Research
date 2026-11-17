"""Check the committed fixture and replay all hand-calculated cases."""

from copy import deepcopy
from fractions import Fraction
from itertools import product
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


sys.path.insert(0, str(ROOT / "src"))

from restore_edge_select.simulator import Experiment
from restore_edge_select.adaptive import AdaptiveConfig, select as adaptive_select


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_adaptive_search():
    # Four candidates: the cheapest prediction is merely the order in which
    # we test; the budget may hide a better, feasible boundary.
    base_row = dict(head_distance=0, prefetch_fits=False, extra_reserve_bytes=0,
                    checkpoint_tier='SSD', queue_debt_ms=0, inflight_ms=0,
                    incremental_ram_ms=0, incremental_ssd_ms=0,
                    head_slack_ms=0, committed_ahead_compute_ms=0)
    rows = {c: dict(base_row, serial_compute_ms=40-c) for c in range(4)}
    floor = {0: (30, 50), 1: (31, 38), 2: (29, 30), 3: (41, 42)}
    ends = {0: (30, 50), 1: (32, 39), 2: (30, 36), 3: (41, 42)}
    rank = {0: 5, 1: 1, 2: 2, 3: 3}
    bounded = adaptive_select(rows, 0, 0, AdaptiveConfig(audit=True), floor, ends.__getitem__, rank)
    require(bounded[:4] == (1, 'light', 12, 9), 'Budgeted search must report a valid nonzero gap')
    full = adaptive_select(rows, 0, 0, AdaptiveConfig(4, 4, 4, audit=True), floor, ends.__getitem__, rank)
    require(full[:4] == (2, 'light', 0, 14), 'Full search must certify the best feasible boundary')
    uncertain = adaptive_select(rows, 0, 0, AdaptiveConfig(
        4, 4, 4, audit=True, forecast_error_ms=3), floor, ends.__getitem__, rank)
    require(uncertain[0] == 0 and uncertain[2] == 0, 'Error margin must reject an uncertain gain')
    screened = adaptive_select(rows, 0, 0, AdaptiveConfig(audit=True, screen=True), floor,
                               ends.__getitem__, {c: c for c in rows})
    require(screened == (0, 'screened-light', None, 0, 14), 'Audit must expose a screening miss')


def check_estimate_guards(fixture, model):
    # A computes until 30 ms. B's shallow/deep candidates need about 67/85 MiB.
    # At 70 MiB only the shallow candidate can prefetch: it finishes at 34.25 ms,
    # while demand-loading the deep candidate finishes at 35 ms. At 128 MiB,
    # the deep candidate can prefetch and finishes at 30 ms instead.
    for cross_tier in (False, True):
        trace = deepcopy(fixture)
        trace["requests"] = trace["requests"][:2]
        if cross_tier:
            next(obj for obj in trace["objects"] if obj["id"] == "B.state1")["tier"] = "RAM"
        for budget in (70, 128):
            exp = Experiment(trace, model, budget)
            baseline = exp.run("local-cost")
            require(baseline["boundaries_tokens"] == [0, 1152]
                    and baseline["completion_ms"] == [30, 35 if budget == 70 else 30],
                    "Guard fixture must retain the deep static boundary and expected completion")
            def classify(rows, base, later):
                require(base == 1152 and rows[576]["prefetch_fits"]
                        and rows[1152]["prefetch_fits"] == (budget == 128)
                        and rows[576]["incremental_ram_ms"] + rows[576]["incremental_ssd_ms"] > 0,
                        "Guard fixture must exercise a prefetched challenger with new transfer work")
                return 4
            opts = dict(screen_mode="classified", screen_selector=classify,
                        score_candidate=lambda row: 0 if row["serial_compute_ms"] == exp.recompute(1, 576) else 1)
            expected = {
                "estimate": (576, 0),
                "estimate-cross-tier": (1152 if cross_tier else 576, 0),
                "estimate-cross-tier-guard": (1152 if cross_tier and budget == 128 else 576,
                                               2 if cross_tier else 0),
                "estimate-prefetch-fallback": (1152 if cross_tier or budget == 70 else 576, 0),
            }
            for mode, (boundary, replays) in expected.items():
                run = exp.run("snapshot-screened", screen_search=mode, **opts)
                completion = [30, Fraction(137, 4)] if boundary == 576 else baseline["completion_ms"]
                require(run["boundaries_tokens"] == [0, boundary]
                        and run["completion_ms"] == completion
                        and run["decision_replays"] == replays,
                        f"{mode}: incorrect guard decision or replay count ({cross_tier=}, {budget=})")


    # Mean improvement alone is insufficient: letting B prefetch a shallow
    # SSD boundary admits C early and helps C/D, but makes B 2.4 ms slower.
    trace = deepcopy(fixture)
    trace["requests"][0]["input_token_runs"][0]["count"] = 7200
    trace["memory"]["execution_mib"] = 512
    trace["cost"]["recompute"]["a_ms_per_token_rational"] = [1, 240]
    trace["cost"]["recompute"]["b_ms_per_pair_rational"] = [0, 1]
    for obj in trace["objects"]:
        if obj["id"] in {"B.state2", "C.state2"}:
            obj["tier"] = "RAM"
    trace["requests"].append(dict(id="D", arrival_ms=1,
        input_token_runs=[dict(token_id=40, count=1152)],
        candidates=[dict(boundary_tokens=0, load_order=[])]))
    exp = Experiment(trace, model, 160)
    opts = dict(screen_mode="classified",
                screen_selector=lambda rows, *_: 4 if rows[1152]["head_distance"] == 1 else None,
                score_candidate=lambda row: 0 if row["serial_compute_ms"] == exp.recompute(1, 576) else 1)
    baseline = exp.run("local-cost")
    proposed = exp.run("snapshot-screened", screen_search="estimate", **opts)
    guarded = exp.run("snapshot-screened", screen_search="estimate-cross-tier-guard", **opts)
    require(proposed["boundaries_tokens"] == [0, 576, 1152, 0]
            and proposed["response_sum_ms"] < baseline["response_sum_ms"]
            and proposed["completion_ms"][1] - baseline["completion_ms"][1] == Fraction(12, 5),
            "Tail guard fixture must improve the sum while slowing B by 2.4 ms")
    require(guarded["boundaries_tokens"] == baseline["boundaries_tokens"]
            and guarded["completion_ms"] == baseline["completion_ms"]
            and guarded["decision_replays"] == 2,
            "Cross-tier guard must reject individual regression even when the queue sum improves")


def main():
    check_adaptive_search()
    fixture = json.loads((ROOT / "fixtures/phase1-hand-trace.json").read_text())
    spec = json.loads((ROOT / "configs/qwen3.5-4b.json").read_text())
    model = spec["models"][0]
    require(fixture["model"] == model["model_id"], "model mismatch")
    require(f"/{fixture['revision']}/" in model["config_url"], "revision mismatch")
    payload = model["payload"]
    objects = {obj["id"]: obj for obj in fixture["objects"]}
    require(len(objects) == len(fixture["objects"]), "duplicate object ID")
    block = fixture["memory"]["kv_block_tokens"]
    arrival = [req["arrival_ms"] for req in fixture["requests"]]
    choices = []
    for req in fixture["requests"]:
        choices.append(set())
        token_run, = req["input_token_runs"]
        for candidate in req["candidates"]:
            c = candidate["boundary_tokens"]
            choices[-1].add(c)
            deps = [objects[obj] for obj in candidate["load_order"]]
            if c == 0:
                require(not deps, f"{req['id']}: full recompute loads objects")
                continue
            require(c % block == 0 and c <= token_run["count"], "invalid boundary")
            require(len(deps) == c // block + 1, "wrong dependency count")
            for end, obj in enumerate(deps[:-1], 1):
                require(obj["kind"] == "attention_kv" and obj["tier"] == "RAM"
                        and obj["start_token_exclusive"] == (end - 1) * block
                        and obj["end_token_inclusive"] == end * block
                        and obj["bytes"] == block * payload["attention_bytes_per_token"],
                        f"invalid KV block: {obj['id']}")
            state = deps[-1]
            require(state["kind"] == "recurrent_and_conv_checkpoint"
                    and state["tier"] == "SSD" and state["boundary_tokens"] == c
                    and state["bytes"] == payload["checkpoint_bytes"], "invalid checkpoint")
            for obj in deps:
                expected = obj.get("end_token_inclusive", c)
                require(obj["prefix_token_runs"] == [
                    {"token_id": token_run["token_id"], "count": expected}
                ], f"prefix mismatch: {obj['id']}")
    combos = set(product(*choices))
    cases = fixture["expected_by_prefetch_mib"]
    require(set(map(int, cases)) == set(fixture["memory"]["prefetch_mib_cases"]),
            "budget cases do not match fixture settings")
    for budget, case in cases.items():
        exp = Experiment(fixture, model, int(budget))
        runs = case["runs"]
        require(len(runs) == len(combos)
                and {tuple(run["boundaries_tokens"]) for run in runs} == combos,
                f"invalid boundary combinations at {budget} MiB")
        for run in runs:
            actual = exp.run("fixed", boundaries=run["boundaries_tokens"])
            for key, expected in run.items():
                require(actual[key] == expected,
                        f"{budget} MiB, {run['boundaries_tokens']}, {key}: "
                        f"{actual[key]} != {expected}")
            finish = run["completion_ms"]
            require(len(finish) == len(arrival) and finish == sorted(finish)
                    and all(f >= a for f, a in zip(finish, arrival)), "invalid completion order")
            require(0 <= run["peak_prefetch_bytes"] <= int(budget) * fixture["units"]["bytes_per_mib"],
                    "invalid prefetch peak")
            require(run["response_sum_ms"] == sum(f - a for f, a in zip(finish, arrival)),
                    "response sum mismatch")
        require(exp.offline()["offline_optima"] == case["offline_optima"], "simulator optimum mismatch")
        best = min(run["response_sum_ms"] for run in runs)
        require({tuple(run["boundaries_tokens"]) for run in runs if run["response_sum_ms"] == best}
                == set(map(tuple, case["offline_optima"])), "offline optimum mismatch")
    exp = Experiment(fixture, model, 160)
    require(exp.run("local-cost")["response_mean_ms"] == 31, "local-cost baseline mismatch")
    require(exp.run("capacity")["response_mean_ms"] == Fraction(123, 4), "capacity baseline mismatch")
    lookahead = exp.run("arrived-lookahead")
    require(lookahead["boundaries_tokens"] == [0, 1152, 576]
            and lookahead["completion_ms"] == [30, 30, Fraction(137, 4)],
            "arrived-lookahead should find the visible joint optimum")
    residual = exp.run("remaining-wait")
    require(residual["boundaries_tokens"] == [0, 1152, 576]
            and residual["completion_ms"] == [30, 30, Fraction(137, 4)],
            "remaining-wait should account for loading hidden by prior compute")
    live = exp.run("remaining-wait-live")
    sampled = Experiment({**fixture, "actual_transfer_ms_by_object": {"B.kv1": 10}}, model, 160)
    sampled_fixed = sampled.run("fixed", boundaries=[0, 1152, 576])
    first = sampled_fixed["transfers"][0]
    require(sampled.transfer_time("B.kv1") == 1 and first["object"] == "B.kv1"
            and first["end_ms"] - first["start_ms"] == 10,
            "sampled service must affect execution without changing the estimator")
    profiled = exp.run("remaining-wait-live", profile_decisions=True)
    require(profiled.pop("decision_profile_ns") and profiled == live, "profiling must not change virtual execution")
    timed = exp.run("remaining-wait-live", inject_measured=True)
    require([end - start for start, end, _ in timed["decision_intervals_ms"]]
            == [Fraction(ns, 1_000_000) for _, ns in timed["decision_profile_ns"]],
            "measured cost must be charged to the dispatcher exactly once")
    variable_speed = Experiment({**fixture, "io_speed_epochs": [(0, 1), ("1.5", 4)]}, model, 160)
    switched = variable_speed.run("fixed", boundaries=[0, 1152, 576])
    require([(t["start_ms"], t["end_ms"]) for t in switched["transfers"][:2]]
            == [(1, 2), (2, Fraction(9, 4))], "speed changes must apply only to new submissions")
    delayed = exp.run("two-request", pair_ms=35)
    require(delayed["decision_intervals_ms"] == [(1, 36, 1)]
            and delayed["completion_ms"] == [30, 41, 46],
            "submitted compute must finish during decision; new dispatch waits")
    require(live["boundaries_tokens"] == residual["boundaries_tokens"]
            and live["completion_ms"] == residual["completion_ms"], "live replay mismatch")
    biased = exp.run("remaining-wait-live", estimate={"ram_to_device": 3, "ssd_to_device": 3})
    fixed = exp.run("fixed", boundaries=biased["boundaries_tokens"])
    require(biased["transfers"] == fixed["transfers"]
            and biased["completion_ms"] == fixed["completion_ms"],
            "forecast errors must not alter execution costs")
    shared = deepcopy(fixture)
    b, c = shared["requests"][1:]
    c["input_token_runs"] = b["input_token_runs"]
    c["candidates"] = b["candidates"]
    shared["cost"]["device_copy"] = {
        "startup_ms": 0, "bytes_per_ms": 2 * payload["checkpoint_bytes"],
    }
    actual = Experiment(shared, model, 128).run("fixed", boundaries=[0, 576, 576])
    require(actual["completion_ms"] == [30, Fraction(139, 4), 39], "shared handoff timing")
    require(actual["peak_prefetch_bytes"] == 70385664, "shared reservation counted twice")
    require(actual["transferred_bytes_by_path"] == {
        "ram_to_device": 18874368, "ssd_to_device": 51511296, "device_copy": 51511296,
    }, "shared dependencies reloaded or checkpoint copy missing")
    # Spread arrivals so the isolated branch is exercised after an epoch
    # change, including a reusable checkpoint whose copy must be charged.
    fast_cases = 0
    for resident, gap, lag, budget in product(
            ((), ("B.state1",), ("B.kv1", "B.state1")),
            (1, 100), (0, 150), (0, 160)):
        scenario = deepcopy(shared)
        for i, request in enumerate(scenario["requests"]):
            request["arrival_ms"] = i * gap
        scenario["scheduling"]["initial_device_object_ids"] = list(resident)
        scenario["io_path_speed_epochs"] = {
            "ram_to_device": [(0, 1), (50, 4)],
            "ssd_to_device": [(0, 1), (75, "1/2")],
        }
        scenario["actual_transfer_ms_by_object"] = {"B.state2": 10}
        exp = Experiment(scenario, model, budget)
        options = dict(estimate={"ram_to_device": 2, "ssd_to_device": "1/2",
                                 "device_copy": 3, "compute": 2}, telemetry_lag_ms=lag)
        live = exp.run("remaining-wait-live", **options)
        fast = exp.run("remaining-wait-fast", **options)
        require(all(live[key] == fast[key] for key in live if key not in {"decision_replays", "decision_events"}),
                "Fast path changed decisions or execution with nonstationary costs")
        if gap == 100:
            require(fast["decision_replays"] == 0 < live["decision_replays"],
                    "Expected the isolated branch to bypass every rollout")
        fast_cases += 1
    slowdown = {"compute": 2, "transfer": 2}
    overlap = {"ram_to_device": slowdown, "ssd_to_device": slowdown}
    exp = Experiment(dict(fixture, overlap_slowdown=overlap), model, 160)
    result = exp.run("fixed", boundaries=[0, 1152, 576])
    require(result["completion_ms"] == [39, 39, Fraction(173, 4)]
            and result["overlap_ms_by_path"] == {"ram_to_device": 6, "ssd_to_device": 12},
            "Compute should consume 9 ms of work in the 18 ms of overlap")
    require([t["end_ms"] for t in result["transfers"]] == [3, 5, 11, 13, 19],
            "Transfers should use the slowed rate while compute remains active")
    short = deepcopy(fixture)
    short["requests"] = short["requests"][:2]
    short["requests"][1]["arrival_ms"] = Fraction(59, 2)
    exp = Experiment(dict(short, overlap_slowdown=overlap), model, 160)
    result = exp.run("fixed", boundaries=[0, 576])
    require(result["completion_ms"] == [Fraction(61, 2), Fraction(153, 4)]
            and [(t["start_ms"], t["end_ms"]) for t in result["transfers"]]
            == [(Fraction(59, 2), 31), (31, 34)],
            "In-flight transfer must return to its base rate when compute ends")
    for budget in (0, 160):
        exp = Experiment(dict(fixture, overlap_slowdown=overlap), model, budget)
        live = exp.run("remaining-wait-live")
        fast = exp.run("remaining-wait-fast")
        fixed = exp.run("fixed", boundaries=live["boundaries_tokens"])
        require(live["boundaries_tokens"] == fast["boundaries_tokens"]
                and live["completion_ms"] == fast["completion_ms"] == fixed["completion_ms"],
                "Overlap forecasts and fixed execution diverged")
        if budget == 0:
            base = Experiment(fixture, model, budget).run("remaining-wait-live")
            require(live["completion_ms"] == base["completion_ms"] and not live["overlap_ms_by_path"],
                    "Without overlap the slowdown model should not affect timing")
    exp = Experiment(fixture, model, 128)
    decisions = []
    def capture(i, now, depth, costs):
        decisions.append(costs)
    observed = exp.run("remaining-wait-live", observe_candidates=capture, intervention_budget_mib=384)
    standard = exp.run("remaining-wait-live")
    require(decisions and all(observed[k] == standard[k] for k in ("completion_ms", "boundaries_tokens", "transfers")),
            "Intervention forecasts must leave the actual run unchanged")
    require(any(len(set(row["intervention_finish_ms"].values())) > 1
                for candidate_set in decisions for row in candidate_set.values()),
            "Intervention fixture did not change any predicted candidate time")
    def score(row):
        require("finish_offset_ms" not in row, "Scorer must not receive forecast outcomes")
        return row["serial_compute_ms"] + row["serial_copy_ms"] + row["serial_io_ms"]
    scored = exp.run("snapshot-score", score_candidate=score)
    require(scored["completion_ms"] == exp.run("local-cost")["completion_ms"],
            "Serial scoring callback must reproduce the local-cost baseline")
    screened = exp.run("snapshot-screened", score_candidate=score, screen_mode="disagree")
    require(screened["completion_ms"] == scored["completion_ms"] and screened["decision_replays"] == 0,
            "Disagreement screen must skip rollouts on identical scores")
    screened = exp.run("snapshot-screened", score_candidate=lambda row: -row["serial_compute_ms"],
                       screen_mode="unsettled", min_gain_ms=1000)
    require(screened["completion_ms"] == scored["completion_ms"] and screened["decision_replays"] > 0,
            "Guard must reject changes below the required forecast gain")
    extra = [{c: 2 * exp.recompute(i, c) for c in choices} for i, choices in enumerate(exp.eligible)]
    snapshots = {}
    def capture_costs(i, now, depth, costs):
        snapshots[i] = costs
    probed = exp.run("remaining-wait-live", observe_candidates=capture_costs,
                     future_cost_cases={"double_future_compute": (extra, {})})
    require(probed["completion_ms"] == standard["completion_ms"]
            and probed["boundaries_tokens"] == standard["boundaries_tokens"],
            "Cost probes must not feed back into the nominal run")
    pair = snapshots[1][0]["future_cost_finish_ms"]
    require(pair["double_future_compute"] - pair["nominal"] == exp.recompute(1, 0),
            "Only future compute submissions, not the active head, should be rescaled")
    exp = Experiment(fixture, model, 160)
    opts = dict(score_candidate=lambda row: row["serial_compute_ms"], screen_mode="unsettled")
    normal = exp.run("snapshot-screened", **opts)
    robust = exp.run("snapshot-screened", future_cost_cases={
        "slower_future_ssd": (exp._suffix_times, {"ssd_to_device": 1000})}, **opts)
    require(normal["boundaries_tokens"] == [0, 1152, 576]
            and robust["boundaries_tokens"] == [0, 1152, 1152]
            and robust["decision_replays"] > normal["decision_replays"],
            "Robust screen should reject a nominal improvement that reverses in a cost probe")
    tiers = Experiment(dict(fixture, transfer_topology="per-tier"), model, 0)
    r = tiers.run("fixed", boundaries=[0, 1152, 576])
    require(r["completion_ms"] == [30, 33, Fraction(161, 4)]
            and [(t["path"], t["start_ms"], t["end_ms"]) for t in r["transfers"][:3]] == [
                ("ram_to_device", 30, 31), ("ssd_to_device", 30, 33), ("ram_to_device", 31, 32)],
            "Separate tier lanes should overlap across tiers but serialize within each tier")
    tiers = Experiment(dict(fixture, transfer_topology="per-tier"), model, 160)
    r = tiers.run("remaining-wait-live")
    require(r["completion_ms"] == tiers.run("fixed", boundaries=r["boundaries_tokens"])["completion_ms"],
            "Per-tier live forecasts must agree with selected-plan replay")
    coupled = dict(fixture, transfer_topology="per-tier",
                   lane_slowdown={"ram_to_device": 2, "ssd_to_device": 2})
    r = Experiment(coupled, model, 0).run("fixed", boundaries=[0, 1152, 576])
    require(r["completion_ms"] == [30, 35, Fraction(173, 4)] and r["lane_overlap_ms"] == 6
            and [(t["path"], t["start_ms"], t["end_ms"]) for t in r["transfers"][:3]] == [
                ("ram_to_device", 30, 32), ("ssd_to_device", 30, 35), ("ram_to_device", 32, 34)],
            "Coupled lanes must conserve completed work and resume solo speed after a peer finishes")
    tiers = Experiment(coupled, model, 160)
    r = tiers.run("remaining-wait-live")
    require(r["completion_ms"] == tiers.run("fixed", boundaries=r["boundaries_tokens"])["completion_ms"],
            "A snapshot of coupled lanes must retain both in-flight remaining work amounts")
    joint = Experiment(dict(coupled, joint_prefill_slowdown={
        "compute": 2, "ram_to_device": 2, "ssd_to_device": 2}), model, 160)
    r = joint.run("fixed", boundaries=[0, 1152, 576])
    require(r["completion_ms"] == [36, 36, Fraction(161, 4)]
            and r["overlap_ms_by_path"]["both_lanes_prefill"] == 12
            and [(t["path"], t["start_ms"], t["end_ms"]) for t in r["transfers"][:2]] == [
                ("ram_to_device", 1, 5), ("ssd_to_device", 1, 13)],
            "Joint prefill must apply each rate only during the three-way overlap")
    r = joint.run("remaining-wait-live")
    require(r["completion_ms"] == joint.run("fixed", boundaries=r["boundaries_tokens"])["completion_ms"],
            "Joint-prefill forecasts must reproduce committed plans")
    for cfg in (fixture, coupled, dict(fixture, overlap_slowdown={
            "ram_to_device": {"compute": 2, "transfer": 2}})):
        search = Experiment(cfg, model, 128)
        opts = dict(score_candidate=lambda r: r["serial_compute_ms"], screen_mode="classified",
                    screen_selector=lambda *_: 4)
        full = search.run("snapshot-screened", **opts)
        for mode in ("lean", "compute", "cutoff", "entry"):
            shorter = search.run("snapshot-screened", **opts, screen_search=mode)
            require(all(shorter[k] == full[k] for k in ("boundaries_tokens", "completion_ms")),
                    f"{mode} changed a candidate comparison in the hand trace")
            if mode != "lean":
                require(shorter["pruned_rollouts"] > 0 and shorter["decision_events"] < full["decision_events"],
                        f"{mode} did not stop losing branches in the hand trace")
    for view in (None, 4):
        estimated = exp.run("snapshot-screened", screen_mode="classified",
                            screen_selector=lambda *_: view, screen_search="estimate",
                            score_candidate=lambda row: -row["serial_compute_ms"])
        expected = exp.run("local-cost")["boundaries_tokens"] if view is None else [0, 0, 0]
        require(estimated["boundaries_tokens"] == expected and estimated["decision_replays"] == 0,
                "Estimate selection must respect the classifier and skip candidate replay")
    opts = dict(score_candidate=lambda r: -r["serial_compute_ms"], screen_mode="unsettled", queue_horizon=4)
    isolated = Experiment(dict(fixture, requests=[fixture["requests"][1]]), model, 128)
    old = isolated.run("snapshot-screened", **opts, screen_search="entry")
    for mode in ("entry-fast", "entry-progress"):
        new = isolated.run("snapshot-screened", **opts, screen_search=mode)
        require(old["decision_replays"] > 0 and new["decision_replays"] == 0
                and old["boundaries_tokens"] == new["boundaries_tokens"]
                and old["completion_ms"] == new["completion_ms"],
                f"{mode} must price an isolated request without a forecast")
    for cfg, estimate in (
        (coupled, None),
        (dict(fixture, staged_checkpoint_cache_mib=64), None),
        (dict(fixture, io_speed_epochs=[(0, 1), (2, "1/2")]), None),
        (fixture, {"ssd_to_device": 2}),
    ):
        special = Experiment(cfg, model, 128)
        old = special.run("snapshot-screened", **opts, estimate=estimate, screen_search="entry")
        for mode in ("entry-fast", "entry-progress"):
            require(special.run("snapshot-screened", **opts, estimate=estimate, screen_search=mode) == old,
                    f"{mode} must fall back when its stationary-cost assumptions do not hold")
    check_estimate_guards(fixture, model)
    for mode, delay, reserved, done in (("start", 0, 30, 35), ("timer", 1, 31, 36),
                                         ("finish", 0, None, Fraction(157, 4))):
        cfg = dict(fixture, scheduling=dict(fixture["scheduling"], prefetch_release=mode, prefetch_hold_ms=delay))
        r = Experiment(cfg, model, 128).run("fixed", boundaries=[0, 576, 1152])
        require(r["prefetch_reserved_at_ms"][2] == reserved and r["completion_ms"][-1] == done,
                "Release timing must constrain later reservations, including mid-service wakeups")
    b = fixture["requests"][1]
    repeat = dict(b, id="B-repeat", arrival_ms=100, candidates=[b["candidates"][0], b["candidates"][-1]])
    for capacity, expected, evictions in ((64, ["ssd_to_device", "ssd_to_device"], 2),
                                           (128, ["ssd_to_device", "ram_to_device"], 0)):
        trace = dict(fixture, staged_checkpoint_cache_mib=capacity, requests=[*fixture["requests"], repeat])
        r = Experiment(trace, model, 0).run("fixed", boundaries=[0, 1152, 576, 1152])
        require([t["path"] for t in r["transfers"] if t["object"] == "B.state2"] == expected
                and r["staged_checkpoint_cache"].get("evictions", 0) == evictions,
                "A retained staging buffer must serve later HBM misses until evicted")
    from restore_edge_select.batch_replay import BatchReplay
    batch = BatchReplay(Experiment(fixture, model, 128), lambda pair: (Fraction(1, 2),) * 2)
    recorded = batch.run(slots=2, fixed=[0, 0, 0])
    require(recorded.done == {0: 46, 1: 17, 2: 33} and recorded.batch_ms == 32,
            "Two slots must conserve solo work as tasks enter and leave at different times")
    plan = [0, 576, 1152]
    one = batch.run(fixed=plan)
    require([one.done[i] for i in range(3)] == batch.exp.run("fixed", boundaries=plan)["completion_ms"]
            and one.reserved[2] == 30,
            "The one-slot model must preserve prefetch release and demand ordering")
    print(f"Checked {len(objects)} objects and {len(cases) * len(combos)} acceptance cases.")
    print(f"Checked {fast_cases} fast-path cases with rate changes and forecast errors.")
    print("Checked dynamic overlap work conservation, boundaries, and no-overlap control.")


if __name__ == "__main__":
    main()
