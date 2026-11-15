"""Measure selection margins in a common queue snapshot and vary offered load."""

import argparse
import csv
from fractions import Fraction as F
from itertools import islice, product
import json
from pathlib import Path
from statistics import mean

from study_bailian import Experiment, ROOT, build, p95, read
from study_boundary_factors import predict
from restore_edge_select.prefill_cost import build_prefill_cost


def diagnose(exp, intervention_budget_mib=None, future_cost_cases=None):
    cases = []

    def observe(i, now, queue_depth, costs):
        def serial(c):
            d = costs[c]
            return d["serial_compute_ms"] + d["serial_io_ms"] + d["serial_copy_ms"]

        def finish(c):
            d = costs[c]
            value = d["finish_offset_ms"]
            if value != d["compute_ms"] + d["head_offset_ms"] + d["exposed_io_and_copy_ms"]:
                raise ValueError("Forecast completion does not match its timing components")
            return value

        isolated = min(costs, key=lambda c: (serial(c), -c))
        live = min(costs, key=lambda c: (finish(c), -c))
        # Compare the two best live choices when there is no change, so
        # that an unchanged winner still has an observable decision margin.
        rival = live if live != isolated else min(
            (c for c in costs if c != isolated), key=lambda c: (finish(c), -c))
        delta = serial(rival) - serial(isolated)
        gain = finish(isolated) - finish(rival)
        reference, alternative = costs[isolated], costs[rival]
        queue_advantage = reference["head_offset_ms"] - alternative["head_offset_ms"]
        def wait_discount(c):
            return c["serial_io_ms"] + c["serial_copy_ms"] - c["exposed_io_and_copy_ms"]
        def extra_compute(c):
            return c["compute_ms"] - c["serial_compute_ms"]
        wait_advantage = wait_discount(alternative) - wait_discount(reference)
        compute_advantage = extra_compute(reference) - extra_compute(alternative)
        assert gain == -delta + queue_advantage + wait_advantage + compute_advantage
        cases.append(dict(request=exp.requests[i]["id"], input_tokens=exp.lengths[i],
                          decision_ms=now, queue_depth=queue_depth,
                          isolated=isolated, live=live, rival=rival, serial_gap_ms=delta,
                          differential_overlap_ms=delta + gain, live_gain_ms=gain,
                          queue_advantage_ms=queue_advantage, wait_advantage_ms=wait_advantage,
                          compute_penalty_advantage_ms=compute_advantage,
                          candidates=costs))

    live = exp.run("remaining-wait-live", observe_candidates=observe,
                   intervention_budget_mib=intervention_budget_mib, future_cost_cases=future_cost_cases)
    assert all(live["boundaries_tokens"][i] == r["live"] for i, r in
               zip((i for i, v in enumerate(exp.eligible) if len(v) > 1), cases, strict=True))
    return live, cases


def reference_ms(exp):
    # This workload has no initial HBM objects and stationary I/O.
    def isolated_io(deps):
        if exp.per_tier:
            return max(sum(exp.transfer_time(o) for o in deps if exp.objects[o]["tier"] == tier)
                       for tier in ("RAM", "SSD"))
        return sum(exp.transfer_time(o) for o in deps)
    return mean(min(exp.recompute(i, c) + isolated_io(deps)
                    for c, deps in candidates.items())
                for i, candidates in enumerate(exp.candidates))


def attribute_first_change(exp, local, approx, scorer):
    a, b = local["boundaries_tokens"], approx["boundaries_tokens"]
    i = next((j for j, (c, d) in enumerate(zip(a, b, strict=True)) if c != d), None)
    if i is None:
        assert local["completion_ms"] == approx["completion_ms"]
        return dict(first_change_index="", attribution={})
    substituted = exp.run("local-cost", decision_overrides={i: b[i]})
    repaired = exp.run("snapshot-score", score_candidate=scorer, decision_overrides={i: a[i]})
    assert substituted["boundaries_tokens"] == [*a[:i], b[i], *a[i + 1:]]
    assert repaired["boundaries_tokens"][:i + 1] == a[:i + 1]

    def effect(with_approx, with_serial):
        delta = [y - x for y, x in zip(with_approx["completion_ms"], with_serial["completion_ms"], strict=True)]
        return dict(earlier_ms=sum(delta[:i]), current_ms=delta[i], later_ms=sum(delta[i + 1:]),
                    mean_ms=mean(delta), later_affected=sum(t != 0 for t in delta[i + 1:]),
                    later_boundary_changes=sum(y != x for y, x in zip(
                        with_approx["boundaries_tokens"][i + 1:], with_serial["boundaries_tokens"][i + 1:], strict=True)))

    explanation = {}
    if approx["response_mean_ms"] > local["response_mean_ms"]:
        def observe(j, now, queue_depth, costs):
            if j == i:
                reference, alternative = costs[a[i]], costs[b[i]]
                explanation.update(
                    decision_ms=now, queue_depth=queue_depth,
                    approx_advantage_ms=scorer(reference) - scorer(alternative),
                    rollout_advantage_ms=reference["finish_offset_ms"] - alternative["finish_offset_ms"],
                    serial_candidate=reference, approx_candidate=alternative,
                )
        exp.run("remaining-wait-live", observe_candidates=observe,
                decision_overrides=dict(enumerate(a[:i])))
        assert explanation and explanation["approx_advantage_ms"] >= 0

    return dict(first_change_index=i, request_id=exp.requests[i]["id"], input_tokens=exp.lengths[i],
                serial_boundary=a[i], approx_boundary=b[i],
                first_snapshot=explanation,
                attribution={"serial_continuation": effect(substituted, local),
                             "approx_continuation": effect(approx, repaired),
                             "full_policy_delta": effect(approx, local)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--windows", type=int, default=6)
    p.add_argument("--skip-windows", type=int, default=0)
    p.add_argument("--max-prompt", type=int, default=2048)
    p.add_argument("--length-split", type=int, default=2048, help="report a cohort above this input length")
    p.add_argument("--placement", choices=("split", "ram", "ssd"), default="split")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--details", type=Path)
    p.add_argument("--grid", type=Path, default=ROOT / "docs/evidence/prefill-vulkan-4k.json")
    p.add_argument("--stability-grid", type=Path,
                   help="probe another measured compute grid and transfer factors 0.9/1/1.1")
    p.add_argument("--io-profile", type=Path, help="JSON with calibrated cost path entries")
    p.add_argument("--io-scale", type=F, default=F(1), help="RAM and SSD transfer-time multiplier")
    p.add_argument("--io-sweep", nargs="+", type=F,
                   help="scan I/O multipliers at measured compute speed and both arrival spacings")
    p.add_argument("--sweep-compute-scale", nargs="+", type=F, default=[F(1)],
                   help="independent continuation-time multipliers for the I/O sweep")
    p.add_argument("--slow-path", choices=("both", "ram", "ssd"), default="both")
    p.add_argument("--transfer-topology", choices=("shared", "per-tier"), default="shared")
    p.add_argument("--zero-only", action="store_true", help="omit wall-clock decision timing")
    p.add_argument("--closed-loop", action="store_true", help="also replay the FIFO snapshot scorer as a policy")
    p.add_argument("--screened", action="store_true", help="compare selective rollout gates and fixed guards")
    p.add_argument("--robust-screen", action="store_true", help="also require improvement across future cost probes")
    p.add_argument("--attribute-first-change", action="store_true",
                   help="swap the first differing boundary under both continuation rules")
    p.add_argument("--intervention-budget", type=int, choices=[384],
                   help="probe 128 -> 384 MiB and active KV handoff on the same snapshots (requires --details)")
    p.add_argument("--overlap-profile", type=Path, help="path-specific compute and I/O slowdown factors")
    p.add_argument("--overlap-model", choices=("none", "static", "dynamic"), default="none")
    args = p.parse_args()
    if (args.windows < 1 or args.skip_windows < 0 or args.max_prompt < 1 or args.length_split < 1
            or min([args.io_scale, *(args.io_sweep or []), *args.sweep_compute_scale]) <= 0):
        p.error("Expected positive window count and I/O scale, and nonnegative skip")
    fixture = read(ROOT / "fixtures/phase1-hand-trace.json")
    model = read(ROOT / "configs/qwen3.5-4b.json")["models"][0]
    grid = read(args.grid)
    io = read(args.io_profile)["cost"] if args.io_profile else {
        "ram_to_device": dict(startup_ms=0, bytes_per_ms=12540000),
        "device_copy": dict(startup_ms=0, bytes_per_ms=112650000),
        "ssd_to_device": dict(startup_ms=0, bytes_per_ms=str(F(51511296 * 1000, 15751))),
    }
    if set(io) != {"ram_to_device", "ssd_to_device", "device_copy"}:
        p.error("Expected RAM, SSD and device copy rates")
    if (args.overlap_model != "none") != bool(args.overlap_profile):
        p.error("Use --overlap-profile with static or dynamic overlap model")
    if args.intervention_budget and not args.details:
        p.error("Interventions require --details")
    if args.transfer_topology == "per-tier" and (args.overlap_model != "none" or not args.zero_only
                                                  or args.closed_loop or args.screened):
        p.error("Per-tier topology uses stationary, zero-overhead exact forecasts without contention")
    if args.robust_screen and not (args.screened and args.stability_grid and args.io_sweep):
        p.error("Robust screening requires --screened, --stability-grid and --io-sweep")
    if args.stability_grid and not (args.robust_screen or (args.details and args.zero_only and args.io_sweep)):
        p.error("Stability probes require --details, --zero-only and --io-sweep")
    args.alternate_compute = (build_prefill_cost(read(args.stability_grid), model) if args.stability_grid else None)
    if args.attribute_first_change and not (args.details and args.closed_loop and args.zero_only):
        p.error("Attribution requires --details, --closed-loop and --zero-only")
    overlap = read(args.overlap_profile) if args.overlap_profile else {}
    manifest = read(ROOT / "data/bailian-manifest.json")
    if args.details:
        args.details.write_text("", encoding="utf-8")
    with args.output.open("w", encoding="utf-8", newline="") as out:
        writer = None
        for entry in manifest["files"]:
            name = entry["name"]
            with (ROOT / "data/raw/bailian" / name).open(encoding="utf-8") as stream:
                for _ in islice(stream, 352 * args.skip_windows):
                    pass
                for w in range(args.skip_windows, args.skip_windows + args.windows):
                    rows = [json.loads(line, parse_float=str) for line in islice(stream, 352)]
                    if len(rows) != 352:
                        raise ValueError(f"Incomplete window: {name}/{w}")
                    trace = build(rows, 256, fixture, model, F(1), args.placement, args.max_prompt)
                    bench_cases(trace, fixture, model, grid, io, overlap, name, w, args, out, writer)
                    writer = True


def bench_cases(trace, fixture, model, grid, io, overlap, name, window, args, out, header_written):
    selected_paths = {"ram_to_device", "ssd_to_device"} if args.slow_path == "both" else {
        "ram_to_device" if args.slow_path == "ram" else "ssd_to_device"}

    def experiment(scale, spacing, budget, io_scale):
        if args.overlap_model == "static":
            scale *= max(F(r["compute"]) for r in overlap.values())
        recompute = {**grid, "samples": [{**s, "ms": [str(F(str(v)) * scale) for v in s["ms"]]}
                                       for s in grid["samples"]]}
        cost = dict(fixture["cost"], recompute=recompute)
        for key, spec in io.items():
            factor = io_scale if key in selected_paths else F(1)
            if args.overlap_model == "static" and key in overlap:
                factor *= F(overlap[key]["transfer"])
            cost[key] = dict(startup_ms=str(F(spec["startup_ms"]) * factor),
                             bytes_per_ms=str(F(spec["bytes_per_ms"]) / factor))
        requests = [dict(r, arrival_ms=r["arrival_ms"] * spacing) for r in trace["requests"]]
        return Experiment(dict(trace, cost=cost, requests=requests,
                               transfer_topology=args.transfer_topology,
                               overlap_slowdown=overlap if args.overlap_model == "dynamic" else {}),
                          model, budget)

    if args.io_sweep:
        conditions = [("io-sweep", scale, spacing, rate)
                      for scale, rate, spacing in product(args.sweep_compute_scale, args.io_sweep,
                                                          (F(1), F(1, 4)))]
    else:
        rate = args.io_scale
        base = reference_ms(experiment(F(1), F(1), 128, rate))
        conditions = [("arrival", F(1), F(1), rate), ("arrival", F(1), F(1, 4), rate)]
        for scale in (F(1, 2), F(2)):
            scaled = reference_ms(experiment(scale, F(1), 128, rate))
            conditions.extend((("fixed-arrival", scale, F(1), rate),
                               ("matched-pressure", scale, scaled / base, rate)))
    for (mode, scale, spacing, rate), budget in product(conditions, (128, 384)):
        exp = experiment(scale, spacing, budget, rate)
        intervention = args.intervention_budget if budget == 128 else None
        probes = None
        if args.alternate_compute:
            alternate = [{c: args.alternate_compute(c, n - c) * scale for c in choices}
                         for n, choices in zip(exp.lengths, exp.eligible, strict=True)]
            probes = {}
            for suffixes, grid_name in ((exp._suffix_times, "nominal"), (alternate, "alternate")):
                for factor in (F(9, 10), F(1), F(11, 10)):
                    if grid_name != "nominal" or factor != 1:
                        probes[f"{grid_name}_io{factor}"] = (suffixes, dict.fromkeys(
                            ("ram_to_device", "ssd_to_device"), factor))
        live, cases = diagnose(exp, intervention, None if args.robust_screen else probes)
        local = exp.run("local-cost")
        if args.overlap_model == "dynamic":
            for policy_run in (live, local, exp.run("remaining-wait-fast")):
                fixed = exp.run("fixed", boundaries=policy_run["boundaries_tokens"])
                assert fixed["completion_ms"] == policy_run["completion_ms"]
        if exp.per_tier:
            for policy_run in (live, local):
                assert exp.run("fixed", boundaries=policy_run["boundaries_tokens"])["completion_ms"] == policy_run["completion_ms"]
        timed_mean, timed_p95 = "", ""
        if not args.zero_only:
            timed = [exp.run(p, inject_measured=True) for p in ("local-cost", "remaining-wait-fast")]
            timed_mean = float(timed[0]["response_mean_ms"] - timed[1]["response_mean_ms"])
            tails = [p95([t - a for t, a in zip(r["completion_ms"], exp.arrivals)]) for r in timed]
            timed_p95 = float(tails[0] - tails[1])
        loop, attribution = {}, {}
        scorer = lambda row: predict("fifo_snapshot", row)
        def tail(run):
            return p95([t - a for t, a in zip(run["completion_ms"], exp.arrivals)])
        if args.closed_loop:
            approx = exp.run("snapshot-score", score_candidate=scorer)
            assert exp.run("fixed", boundaries=approx["boundaries_tokens"])["completion_ms"] == approx["completion_ms"]
            if args.attribute_first_change:
                attribution = attribute_first_change(exp, local, approx, scorer)
            loop = dict(
                approx_vs_serial_mean_ms=float(local["response_mean_ms"] - approx["response_mean_ms"]),
                approx_vs_serial_p95_ms=float(tail(local) - tail(approx)),
                approx_vs_live_mean_ms=float(live["response_mean_ms"] - approx["response_mean_ms"]),
                approx_vs_live_p95_ms=float(tail(live) - tail(approx)),
                approx_changes_vs_serial=sum(a != b for a, b in
                                             zip(approx["boundaries_tokens"], local["boundaries_tokens"])),
                approx_changes_vs_live=sum(a != b for a, b in
                                           zip(approx["boundaries_tokens"], live["boundaries_tokens"])),
            )
            if not args.zero_only:
                timed_approx = exp.run("snapshot-score", score_candidate=scorer, inject_measured=True)
                loop.update(
                    timed_approx_vs_serial_mean_ms=float(timed[0]["response_mean_ms"] - timed_approx["response_mean_ms"]),
                    timed_approx_vs_serial_p95_ms=float(tail(timed[0]) - tail(timed_approx)),
                )
        if args.screened:
            modes = [(f"{m}_{t}", dict(screen_mode=m, min_gain_ms=t))
                     for m, t in product(("disagree", "unsettled"), (0, 30))]
            if args.robust_screen:
                modes.append(("robust", dict(screen_mode="unsettled", future_cost_cases=probes)))
            for prefix, opts in modes:
                opts["score_candidate"] = scorer
                screened = exp.run("snapshot-screened", **opts)
                assert exp.run("fixed", boundaries=screened["boundaries_tokens"])["completion_ms"] == screened["completion_ms"]
                loop.update({
                    f"{prefix}_mean_gain_ms": float(local["response_mean_ms"] - screened["response_mean_ms"]),
                    f"{prefix}_p95_gain_ms": float(tail(local) - tail(screened)),
                    f"{prefix}_replays": screened["decision_replays"],
                    f"{prefix}_choices_vs_live": sum(a != b for a, b in zip(
                        screened["boundaries_tokens"], live["boundaries_tokens"], strict=True)),
                })
                if not args.zero_only:
                    wall = exp.run("snapshot-screened", **opts, inject_measured=True)
                    loop.update({
                        f"{prefix}_timed_mean_gain_ms": float(timed[0]["response_mean_ms"] - wall["response_mean_ms"]),
                        f"{prefix}_timed_p95_gain_ms": float(tail(timed[0]) - tail(wall)),
                        f"{prefix}_timed_cpu_ms": sum(t for _, t in wall["decision_profile_ns"]) / 1e6,
                    })
            loop["live_replays"] = live["decision_replays"]
            if not args.zero_only:
                loop["timed_fast_cpu_ms"] = sum(t for _, t in timed[1]["decision_profile_ns"]) / 1e6
        flips = [r for r in cases if r["isolated"] != r["live"]]
        span = exp.arrivals[-1] - exp.arrivals[0]
        row = dict(trace=name, window=window, placement=args.placement, mode=mode, scale=str(scale),
                   max_prompt=args.max_prompt,
                   spacing=float(spacing), budget_mib=budget, io_scale=str(rate),
                   slow_path=args.slow_path,
                   transfer_topology=args.transfer_topology,
                   overlap_model=args.overlap_model,
                   pressure=float(reference_ms(exp) * (len(exp.arrivals) - 1) / span) if span else "",
                   requests=len(exp.requests), competing=len(cases),
                   requests_over_2048=sum(n > 2048 for n in exp.lengths),
                   competing_over_2048=sum(r["input_tokens"] > 2048 for r in cases),
                   strict_flips_over_2048=sum(r["input_tokens"] > 2048 and r["live_gain_ms"] > 0
                                              for r in flips),
                   length_split=args.length_split,
                   requests_over_split=sum(n > args.length_split for n in exp.lengths),
                   competing_over_split=sum(r["input_tokens"] > args.length_split for r in cases),
                   strict_flips_over_split=sum(r["input_tokens"] > args.length_split and r["live_gain_ms"] > 0
                                                for r in flips),
                   near_tie_1ms=sum(abs(r["live_gain_ms"]) <= 1 for r in cases),
                   any_discount=sum(any(c["serial_io_ms"] + c["serial_copy_ms"] > c["exposed_io_and_copy_ms"]
                                        for c in r["candidates"].values()) for r in cases),
                   minimum_margin_ms=float(min(abs(r["live_gain_ms"]) for r in cases)) if cases else "",
                   positive_relative_overlap=sum(r["differential_overlap_ms"] > 0 for r in cases),
                   flips=len(flips), strict_flips=sum(r["live_gain_ms"] > 0 for r in flips),
                   tie_flips=sum(r["live_gain_ms"] == 0 for r in flips),
                   matched_flip_gain_ms=float(sum(r["live_gain_ms"] for r in flips)),
                   competing_serial_recompute=sum(r["isolated"] == 0 for r in cases),
                   competing_live_recompute=sum(r["live"] == 0 for r in cases),
                   flips_to_recompute=sum(r["live"] == 0 for r in flips),
                   flips_from_recompute=sum(r["isolated"] == 0 for r in flips),
                   mean_queue_ms=float(mean(live["queue_wait_ms"])),
                   live_mean_ms=float(live["response_mean_ms"]),
                   live_p95_ms=float(p95([t - a for t, a in zip(live["completion_ms"], exp.arrivals)])),
                   ram_overlap_ms=float(live.get("overlap_ms_by_path", {}).get("ram_to_device", 0)),
                   ssd_overlap_ms=float(live.get("overlap_ms_by_path", {}).get("ssd_to_device", 0)),
                   zero_mean_gain_ms=float(local["response_mean_ms"] - live["response_mean_ms"]),
                   timed_mean_gain_ms=timed_mean, timed_p95_gain_ms=timed_p95, **loop)
        writer = csv.DictWriter(out, fieldnames=row, lineterminator="\n")
        if not header_written:
            writer.writeheader()
            header_written = True
        writer.writerow(row)
        if args.details:
            with args.details.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"scenario": row, "decisions": cases,
                                         **({"first_change": attribution} if args.attribute_first_change else {})},
                                        default=float) + "\n")


if __name__ == "__main__":
    main()
