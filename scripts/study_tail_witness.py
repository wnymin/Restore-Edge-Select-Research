"""Identify requests affected by the first boundary change in a Bailian window."""

import argparse
from bisect import bisect_right
from fractions import Fraction as F
from itertools import islice
import json
from pathlib import Path
from statistics import mean

from study_bailian import Experiment, ROOT, build, p95, read
from study_boundary_factors import predict
from study_load_routing import ESTIMATE_POLICIES, PRESSURE_POLICIES, pressure_view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    manifest = read(ROOT / "data/bailian-manifest.json")
    parser.add_argument("--trace", choices=[entry["name"] for entry in manifest["files"]],
                        default="qwen_coder_blksz_16.jsonl")
    parser.add_argument("--window", type=int, default=37)
    parser.add_argument("--placement", choices=("split", "ram", "ssd"), default="split")
    parser.add_argument("--spacing", type=F, default=F(1))
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--rate", type=F, default=F(16))
    parser.add_argument("--path", choices=("ssd", "all"), default="ssd")
    parser.add_argument("--max-prompt", type=int, default=6656)
    parser.add_argument("--grid", type=Path, default=ROOT / "docs/evidence/prefill-vulkan-8k.json")
    parser.add_argument("--overlap-profile", type=Path, default=ROOT / "configs/overlap-observed-upper.json")
    parser.add_argument("--ssd-estimate-factor", type=F, default=F(1))
    parser.add_argument("--intervene-policy", choices=("entry", *PRESSURE_POLICIES),
                        default="mixed-reuse-estimate")
    parser.add_argument("--reference-policy", choices=("local-cost", *PRESSURE_POLICIES),
                        default="local-cost")
    args = parser.parse_args()
    window, trace = args.window, args.trace
    with (ROOT / "data/raw/bailian" / trace).open(encoding="utf-8") as stream:
        rows = [json.loads(line, parse_float=str)
                for line in islice(stream, 352 * window, 352 * (window + 1))]
    if len(rows) != 352:
        raise ValueError("Tail-witness window is incomplete")
    fixture = read(ROOT / "fixtures/phase1-hand-trace.json")
    model = read(ROOT / "configs/qwen3.5-4b.json")["models"][0]
    data = build(rows, 256, fixture, model, args.spacing, args.placement, args.max_prompt)
    grid = read(args.grid)
    io = read(ROOT / "docs/evidence/load-staged-ssd-profile.json")["cost"]
    cost = dict(fixture["cost"], recompute=grid)
    for path, config in io.items():
        scale = args.rate if path == "ssd_to_device" or (args.path == "all" and path == "ram_to_device") else 1
        cost[path] = dict(startup_ms=str(F(config["startup_ms"]) * scale),
                          bytes_per_ms=str(F(config["bytes_per_ms"]) / scale))
    overlap = read(args.overlap_profile)
    if args.ssd_estimate_factor <= 0 or (overlap and args.ssd_estimate_factor != 1):
        parser.error("SSD estimate factor must be positive and requires empty overlap profile when varied")
    estimate = ({"ssd_to_device": args.ssd_estimate_factor}
                if args.ssd_estimate_factor != 1 else None)
    exp = Experiment(dict(data, cost=cost, overlap_slowdown=overlap), model, args.budget)
    opts = dict(score_candidate=lambda row: predict("fifo_snapshot", row),
                screen_mode="unsettled", queue_horizon=4, screen_search="entry", estimate=estimate)
    base = exp.run("local-cost", estimate=estimate)
    queue = exp.run("snapshot-screened", **opts)
    pressure_runs = {}
    intervention_options = opts
    for label, allow_reuse in PRESSURE_POLICIES.items():
        pressure_options = dict(
            score_candidate=opts["score_candidate"], screen_mode="classified", estimate=estimate,
            screen_search=ESTIMATE_POLICIES.get(label, "entry"),
            screen_selector=lambda rows, base, _later, enabled=allow_reuse:
                pressure_view(rows, base, allow_ready_reuse=enabled),
        )
        pressure_runs[label] = exp.run("snapshot-screened", **pressure_options)
        if label == args.intervene_policy:
            intervention_options = pressure_options
    selected_run = pressure_runs.get(args.intervene_policy, queue)
    if args.reference_policy != "local-cost":
        base = pressure_runs[args.reference_policy]
    baseline, selected = base["boundaries_tokens"], selected_run["boundaries_tokens"]
    changes = [i for i, (a, b) in enumerate(zip(baseline, selected, strict=True)) if a != b]

    def response(run):
        return [t - a for t, a in zip(run["completion_ms"], exp.arrivals, strict=True)]

    def compare(run, reference=base):
        old, new = response(reference), response(run)
        return dict(mean_gain_ms=float(mean(old) - mean(new)), p95_gain_ms=float(p95(old) - p95(new)))

    def intervention(i):
        decision_time = None
        decision_signals = {}

        def record(j, now, _depth, costs):
            nonlocal decision_time
            if j == i:
                decision_time = now
                for boundary, row in costs.items():
                    decision_signals[boundary] = {
                        key: float(row[key]) if isinstance(row[key], F) else row[key]
                        for key in ("checkpoint_tier", "prefetch_fits", "serial_compute_ms",
                                    "incremental_ram_ms", "incremental_ssd_ms", "missing_bytes",
                                    "ready_bytes", "shared_reservation_bytes", "active_new_kv_bytes")
                    }
                    decision_signals[boundary]["predicted_fifo_ms"] = float(predict("fifo_snapshot", row))

        exp.run("remaining-wait-live", decision_overrides=dict(enumerate(baseline[:i])),
                observe_candidates=record, estimate=estimate)
        if decision_time is None:
            raise RuntimeError(f"No observed decision for request {i}")
        visible = bisect_right(exp.arrivals, decision_time)
        end_of_horizon = min(visible, i + 5)
        fixed = exp.run("fixed", boundaries=[*baseline[:i], selected[i], *baseline[i + 1:]])
        undo = exp.run("snapshot-screened", **intervention_options, decision_overrides={i: baseline[i]})
        if i == changes[0] and undo["boundaries_tokens"][:i + 1] != baseline[:i + 1]:
            raise RuntimeError("First-choice reversal did not reproduce the baseline prefix")
        old = response(base)
        changed = [dict(request_id=base["request_ids"][j], position=j,
                        arrived=j < visible, in_horizon=j < end_of_horizon,
                        arrival_ms=float(exp.arrivals[j]), baseline_ms=float(a), intervention_ms=float(b),
                        delta_ms=float(b - a))
                   for j, (a, b) in enumerate(zip(old, response(fixed), strict=True)) if a != b]
        request_id = base["request_ids"][i]
        transfers = {
            label: [dict(path=t["path"], mode=t["mode"], object=t["object"], bytes=t["bytes"],
                         start_ms=float(t["start_ms"]), end_ms=float(t["end_ms"]))
                    for t in run["transfers"] if t["request"] == request_id]
            for label, run in (("baseline", base), ("changed", fixed))
        }
        return dict(request_id=request_id, position=i, decision_ms=float(decision_time),
                    baseline_boundary=baseline[i], selected_boundary=selected[i],
                    visible_requests=visible, horizon_end_exclusive=end_of_horizon,
                    only_this_change=compare(fixed), revert_in_policy=compare(undo),
                    affected_requests=changed, transfers=transfers,
                    decision_signals=decision_signals)

    def isolated_gain(i):
        changed = exp.run("fixed", boundaries=[*baseline[:i], selected[i], *baseline[i + 1:]])
        return base["response_mean_ms"] - changed["response_mean_ms"]

    if args.reference_policy != "local-cost" and changes:
        gains = {i: isolated_gain(i) for i in changes}
        best_change = max(gains, key=gains.get)
        comparison = dict(
            isolated_mean_gain_ms={base["request_ids"][i]: float(gain) for i, gain in gains.items()},
            strongest_isolated_change=intervention(best_change),
        )
    else:
        comparison = {}

    controls = {}
    for label, paths in (("ram_only", ("ram_to_device",)),
                         ("ssd_only", ("ssd_to_device",)),
                         ("both", tuple(overlap))):
        # Disable the selected compute slowdowns. Keep transfer slowdown
        # and both boundary vectors fixed in each paired control.
        control_overlap = {
            path: dict(config, compute="1") if path in paths else config
            for path, config in overlap.items()
        }
        control = Experiment(dict(data, cost=cost, overlap_slowdown=control_overlap), model, args.budget)
        controls[label] = compare(control.run("fixed", boundaries=selected),
                                  control.run("fixed", boundaries=baseline))
    result = dict(window=window, trace=trace, max_prompt=args.max_prompt, placement=args.placement,
                  spacing=float(args.spacing), budget_mib=args.budget, rate=float(args.rate),
                  path=args.path, ssd_estimate_factor=float(args.ssd_estimate_factor),
                  intervene_policy=args.intervene_policy, reference_policy=args.reference_policy,
                  request_count=len(baseline), selected_policy=compare(selected_run),
                  queue=compare(queue),
                  pressure_policies={label: dict(**compare(run), replays=run["decision_replays"],
                                                 boundary_at_first_change=run["boundaries_tokens"][changes[0]]
                                                 if changes else None)
                                     for label, run in pressure_runs.items()},
                  fixed_plans_without_compute_slowdown=controls,
                  changed_boundaries=len(changes),
                  first_change=intervention(changes[0]) if changes else None, **comparison)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
