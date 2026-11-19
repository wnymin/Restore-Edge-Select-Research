"""Build the R02 CPU development reference; not a holdout or a state reduction."""

import argparse
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path

from scripts.cpu.lookahead.robustness.study_lookahead_first_decision import Capture
from scripts.cpu.lookahead.screening.study_lookahead_phase1 import ROOT, read
from scripts.cpu.lookahead.staging.study_lookahead_dynamic_release import make_trace
from scripts.cpu.lookahead.staging.study_lookahead_shared_projection import shared_trace
from scripts.cpu.lookahead.staging.study_lookahead_three_request import seed_trace
from scripts.cpu.lookahead.staging.study_lookahead_two_request import jsonable
from src.restore_edge_select.simulator import Experiment


MIB = 1 << 20
PHASE = ("now_ms", "head_index", "visible_count", "admitted", "ready", "compute_end_ms",
         "transfer", "active_dependencies", "held_reservation", "release_ms", "host_cache")


def state_record(snap):
    record = dict(zip(PHASE, snap, strict=True))
    record["admitted"] = {i: sorted(objects) for i, objects in record["admitted"].items()}
    for key in ("ready", "held_reservation"):
        record[key] = sorted(record[key])
    return record


def inputs():
    # Cross an exact admission threshold under two identity layouts and costs.
    for layout in ("distinct", "shared-kv", "shared-checkpoint"):
        trace = seed_trace() if layout == "distinct" else shared_trace()
        if layout == "shared-checkpoint":
            trace["cost"]["device_copy"] = dict(startup_ms=0, bytes_per_ms=16 * MIB)
            trace["objects"] = [o for o in trace["objects"] if not o["id"].startswith("C.state")]
            for c in trace["requests"][2]["candidates"]:
                c["load_order"] = [o.replace("C.state", "B.state") for o in c["load_order"]]
        for cost in (1, 2):
            for slack in (-1, 0, 1):
                t = json.loads(json.dumps(trace))
                t["cost"]["ssd_to_device"]["bytes_per_ms"] = 17170432 // cost
                yield f"{layout}-ssd{cost}-slack{slack:+d}", t, (0, 0, 1152), (0, 1152), slack, None
    # For the 145 MiB case with arm 576: B.kv1 takes 1 ms; B.state1 and
    # D.state1 have the same cost, so D readiness is 1 + 2*s. A ends at 30.
    for offset in (F(-1, 4), F(0), F(1, 4)):
        t = make_trace()
        t["cost"]["ssd_to_device"]["bytes_per_ms"] = str(2 * 51511296 / (29 + offset))
        yield f"release-ready-{offset}", t, (0, 576, 1152, 576), (0, 576), None, 145 * MIB
    t = make_trace()
    yield "head-demand", t, (0, 576, 1152, 576), (0, 576), None, 0
    t = make_trace()
    t["cost"]["ssd_to_device"]["bytes_per_ms"] = str(F(51511296) / F(11, 4))
    yield "head-demand-exact-tie", t, (0, 576, 1152, 576), (0, 576), None, 0
    t = seed_trace()
    t["cost"]["device_copy"] = dict(startup_ms=0, bytes_per_ms=16 * MIB)
    t["scheduling"]["initial_device_object_ids"] = [o["id"] for o in t["objects"]]
    yield "all-resident-control", t, (0, 0, 1152), (0, 1152), None, 0


def shared_release_inputs():
    """Match C/D prompts, boundary and costs; vary identity and fixed quota only."""
    for layout in ("distinct", "shared-kv", "shared-checkpoint"):
        trace = make_trace()
        trace["scheduling"]["initial_device_object_ids"] = []
        trace["cost"]["device_copy"] = dict(startup_ms=0, bytes_per_ms=16 * MIB)
        _, _, c, d = trace["requests"]
        c["candidates"] = deepcopy([arm for arm in d["candidates"] if arm["boundary_tokens"] in (0, 576)])
        for arm in c["candidates"]:
            arm["load_order"] = [obj.replace("D.", "C.") for obj in arm["load_order"]]
        d["input_token_runs"] = deepcopy(c["input_token_runs"])
        for obj in trace["objects"]:
            if obj["id"].startswith("D."):
                for run in obj.get("prefix_token_runs", []):
                    run["token_id"] = c["input_token_runs"][0]["token_id"]
        for arm in d["candidates"]:
            if layout != "distinct":
                arm["load_order"] = [obj.replace("D.kv", "C.kv") for obj in arm["load_order"]]
            if layout == "shared-checkpoint":
                arm["load_order"] = [obj.replace("D.state", "C.state") for obj in arm["load_order"]]
        for mib in (110, 120):
            yield f"release-sharing-{layout}-{mib}", deepcopy(trace), (0, 576, 576, 576), (0, 576), None, mib*MIB


def reference(*, shared_release=False):
    model = read(ROOT / "configs/qwen3.5-4b.json")["models"][0]
    rows = []
    protocol = "R02-CPU-01/dev-release-sharing-v1" if shared_release else "R02-CPU-01/dev-v1"
    for name, trace, base, options, slack, capacity in (shared_release_inputs() if shared_release else inputs()):
        trace["fixture_id"] = f"r02-cpu-dev:{name}"
        trace["purpose"] = "Synthetic R02 development input, not an independent holdout."
        probe = Experiment(trace, model, 0)
        threshold = None
        if slack is not None:
            # Union, not summed object sizes, determines the deep arm threshold.
            occupied = set(probe.candidates[1][1152]) | set(probe.candidates[2][1152])
            threshold = probe.size(occupied - probe.initial)
            capacity = threshold + slack
        exp = Capture(trace, model, F(capacity, MIB))
        exp.run_trigger({}, "full")
        target, fallback, snap, _, estimate, speeds = exp.records[0]
        assert (target, snap[0], snap[1], snap[2]) == (1, 0, 0, len(base))
        assert all(arrival == snap[0] for arrival in exp.arrivals)
        assert not (exp.per_tier or exp.dynamic or exp.host_cache_bytes) and exp.release_at == "start"
        arms = {}
        for c in options:
            plan = (base[0], c, *base[2:])
            args = (exp, "fixed", 0, plan, 100, len(base))
            opts = dict(estimate=estimate, forecast_speed=speeds)
            full = Experiment._run(*args, snapshot=deepcopy(snap), **opts)
            cut = Experiment._run(*args, snapshot=deepcopy(snap), **opts, lean_rollout=True, pause_at_head=1)
            resumed = Experiment._run(*args, snapshot=deepcopy(cut["snapshot"]), **opts)
            direct = exp.run("fixed", boundaries=plan)
            assert direct["completion_ms"] == full["completion_ms"] == cut["completion_ms"] + resumed["completion_ms"], (name, c)
            arms[c] = dict(plan=plan, completion_ms=full["completion_ms"],
                           response_sum_ms=direct["response_sum_ms"], response_mean_ms=direct["response_mean_ms"],
                           events=direct["transfers"], prefetch_reserved_ms=direct["prefetch_reserved_at_ms"],
                           compute_start_ms=[a + h + t for a,h,t in zip(exp.arrivals, direct["queue_wait_ms"],
                                                                       direct["head_to_compute_ms"])],
                           next_head_state=state_record(cut["snapshot"]),
                           completed_before_resume_ms=cut["completion_ms"])
        if slack is not None:
            assert (arms[1152]["prefetch_reserved_ms"][2] == 0) == (slack >= 0), name
        if name.startswith("release-ready-"):
            out = arms[576]
            end = next(r["end_ms"] for r in out["events"] if r["object"] == "D.state1")
            assert end - out["compute_start_ms"][1] == F(name.removeprefix("release-ready-")), name
            active = out["next_head_state"]["transfer"]
            if F(name.removeprefix("release-ready-")) > 0:
                assert active == (end, "D.state1"), (name, active)
            else:
                assert active is None, (name, active)
        if name == "all-resident-control":
            assert all(not r["events"] for r in arms.values())
        if name.startswith("head-demand"):
            assert all(not any(x is not None for x in r["prefetch_reserved_ms"]) for r in arms.values())
            assert all(r["events"] and all(x["mode"] == "demand" for x in r["events"]) for r in arms.values())
        if shared_release:
            charged = arms[576]
            assert charged["prefetch_reserved_ms"][1:3] == [0, 30], name
            assert charged["compute_start_ms"][1] == 30 and arms[0]["prefetch_reserved_ms"][2] == 0, name
            union = set(exp.candidates[2][576]) | set(exp.candidates[3][576])
            expected = F(30) if exp.size(union) <= capacity else charged["compute_start_ms"][2]
            assert charged["prefetch_reserved_ms"][3] == expected, name
            objects = [x["object"] for x in charged["events"] if x["mode"] == "prefetch"]
            assert ("D.kv1" in objects) == ("distinct" in name), name
            assert ("D.state1" in objects) == ("shared-checkpoint" not in name), name
            assert any(x["mode"] == "handoff" for x in charged["events"]) == ("shared-checkpoint" in name), name
        best = min(r["response_sum_ms"] for r in arms.values())
        minima = sorted(c for c in arms if arms[c]["response_sum_ms"] == best)
        if name == "head-demand-exact-tie":
            assert minima == [0, 576]
        # Report all exact minima separately from deterministic (score, -c)
        # ordering, and record native fallback without applying it to a subset.
        rows.append(dict(case_id=f"{protocol}:{name}", trace=trace,
                         capacity_bytes=capacity, threshold_bytes=threshold,
                         request_ids=[r["id"] for r in exp.requests], target_index=target,
                         original_fallback=fallback,
                         initial_snapshot=state_record(snap), arms=arms,
                         exact_minima=minima, score_order_pick=max(minima)))
    return jsonable(dict(protocol=protocol, split="development-only",
                         model_config="configs/qwen3.5-4b.json", cases=rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", type=Path)
    action.add_argument("--check", type=Path)
    p.add_argument("--shared-release", action="store_true", help="Use the matched sharing/release development matrix")
    args = p.parse_args()
    result = reference(shared_release=args.shared_release)
    if args.write:
        with args.write.open("x", encoding="utf-8") as f:
            # One complete case per line keeps exact event records diffable.
            header = {k: v for k, v in result.items() if k != "cases"}
            f.write(json.dumps(header, ensure_ascii=False)[:-1] + ', "cases": [\n')
            f.write(",\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                               for row in result["cases"]))
            f.write("\n]}\n")
    elif result != read(args.check):
        p.exit(1, "R02 development reference mismatch\n")
    print(f"Checked {len(result['cases'])} snapshots and {sum(len(r['arms']) for r in result['cases'])} fixed arms")


if __name__ == "__main__":
    main()
