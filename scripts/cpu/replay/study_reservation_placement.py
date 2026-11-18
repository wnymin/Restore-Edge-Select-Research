"""Paired counterfactuals for reservation release and retained SSD staging buffers."""

import argparse
import csv
from fractions import Fraction as F
from itertools import islice, product
import json
from pathlib import Path
from statistics import mean

from scripts.cpu.replay.study_bailian import Experiment, ROOT, build, p95, read
from scripts.cpu.replay.study_boundary_factors import predict


def evaluate(exp, base_fields, writer, timed):
    lifetime = dict(candidates=0, blocked=0, blocked_active=0, admitted=0, same_source=0,
                    new_source=0, immediate_handoff=0, same_source_witness="")

    def observe(i, now, depth, costs):
        for c, d in costs.items():
            if not c or d["head_distance"] == 0:
                continue
            lifetime["candidates"] += 1
            if d["prefetch_fits"]:
                lifetime["immediate_handoff"] += bool(d["active_kv_handoff_bytes"])
                continue
            lifetime["blocked"] += 1
            lifetime["blocked_active"] += d["active_new_kv_bytes"] > 0
            t = d["prefetch_reservation_offset_ms"]
            if t is None:
                continue
            assert 0 <= t <= d["head_offset_ms"]
            lifetime["admitted"] += 1
            source = d["active_kv_handoff_source"]
            if source is not None:
                key = "same_source" if d["active_source_index"] == source else "new_source"
                lifetime[key] += 1
                if key == "same_source" and not lifetime["same_source_witness"]:
                    lifetime["same_source_witness"] = json.dumps(dict(request=i, boundary=c, source=source,
                        decision_ms=float(now), delay_ms=float(t), bytes=d["active_kv_handoff_bytes"]))

    live = exp.run("remaining-wait-live", observe_candidates=observe)
    base = exp.run("local-cost")
    if timed:
        wall_base = exp.run("local-cost", inject_measured=True)
    opts = dict(screen_mode="unsettled", score_candidate=lambda row: predict("fifo_snapshot", row))
    probes = {str(f): (exp._suffix_times, dict.fromkeys(("ram_to_device", "ssd_to_device"), f))
              for f in (F(9, 10), F(11, 10))}
    for policy, extra in (("single", {}), ("next4", {"queue_horizon": 4}),
                          ("guard4", {"queue_horizon": 4, "future_cost_cases": probes})):
        run = exp.run("snapshot-screened", **(opts | extra))
        repeat = exp.run("fixed", boundaries=run["boundaries_tokens"])
        assert (run["completion_ms"], run.get("staged_checkpoint_cache")) == (
            repeat["completion_ms"], repeat.get("staged_checkpoint_cache"))
        tail = lambda r: p95([t - a for t, a in zip(r["completion_ms"], exp.arrivals, strict=True)])
        eligible = [(i, b) for i, (b, a) in enumerate(zip(run["boundaries_tokens"], base["boundaries_tokens"],
                                                        strict=True)) if a != b]
        first = earlier = current = later = ""
        if eligible and run["response_mean_ms"] > base["response_mean_ms"]:
            first, change = eligible[0]
            intervention = exp.run("local-cost", decision_overrides={first: change})
            diff = [float(a - b) for a, b in zip(intervention["completion_ms"], base["completion_ms"], strict=True)]
            earlier, current, later = sum(diff[:first]), diff[first], sum(diff[first + 1:])
        wall = exp.run("snapshot-screened", **(opts | extra), inject_measured=True) if timed else None
        stats = run.get("staged_checkpoint_cache", {})
        writer.writerow(dict(**base_fields, **lifetime, requests=len(exp.requests), policy=policy,
                             base_mean_ms=float(base["response_mean_ms"]),
                             mean_gain_ms=float(base["response_mean_ms"] - run["response_mean_ms"]),
                             p95_gain_ms=float(tail(base) - tail(run)),
                             timed_gain_ms=float(wall_base["response_mean_ms"] - wall["response_mean_ms"]) if wall else "",
                             timed_p95_gain_ms=float(tail(wall_base) - tail(wall)) if wall else "",
                             cpu_ms=sum(v for _, v in wall["decision_profile_ns"]) / 1e6 if wall else "",
                             host_hits=stats.get("hits", 0), host_evictions=stats.get("evictions", 0),
                             live_changed=sum(a != b for a, b in zip(live["boundaries_tokens"],
                                                                    base["boundaries_tokens"], strict=True)),
                             policy_changed=len(eligible), first_change=first, earlier_ms=earlier,
                             current_ms=current, later_ms=later))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", type=int, choices=(18, 24), required=True)
    p.add_argument("--windows", type=int, default=6)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--zero-only", action="store_true")
    p.add_argument("--timer-ms", type=int, help="run only the bounded reservation hold")
    p.add_argument("--trace", help="restrict a mechanism check to one manifest file")
    p.add_argument("--grid", type=Path, default=ROOT / "docs/evidence/prefill-vulkan-8k.json",
                   help="Continuation cost grid; default reproduces the archived experiment")
    p.add_argument("--io-profile", type=Path, default=ROOT / "docs/evidence/load-staged-ssd-profile.json",
                   help="Transfer cost profile; default reproduces the archived experiment")
    args = p.parse_args()
    if args.windows < 1 or (args.timer_ms is not None and args.timer_ms <= 0):
        p.error("Expected a positive window count and release delay")
    cap = 4096 if args.cohort == 18 else 8192
    fixture = read(ROOT / "fixtures/phase1-hand-trace.json")
    model = read(ROOT / "configs/qwen3.5-4b.json")["models"][0]
    grid = read(args.grid)
    io = read(args.io_profile)["cost"]
    overlap = read(ROOT / "configs/overlap-observed-upper.json")
    sources = [e for e in read(ROOT / "data/bailian-manifest.json")["files"] if
               not args.trace or e["name"] == args.trace]
    if not sources:
        p.error("Unknown manifest file")
    fields = ("cohort", "trace", "window", "spacing", "budget_mib", "io_scale", "release", "host_mib", "policy",
              "requests", "base_mean_ms", "mean_gain_ms", "p95_gain_ms", "timed_gain_ms", "timed_p95_gain_ms",
              "cpu_ms", "policy_changed", "live_changed", "candidates", "blocked", "blocked_active", "admitted",
              "same_source", "new_source", "immediate_handoff", "same_source_witness", "host_hits", "host_evictions",
              "first_change", "earlier_ms", "current_ms", "later_ms")
    with args.out.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fields, lineterminator="\n")
        writer.writeheader()
        for entry in sources:
            trace = entry["name"]
            with (ROOT / "data/raw/bailian" / trace).open(encoding="utf-8") as stream:
                for _ in islice(stream, 352 * args.cohort):
                    pass
                for window in range(args.cohort, args.cohort + args.windows):
                    rows = [json.loads(line, parse_float=str) for line in islice(stream, 352)]
                    if len(rows) != 352:
                        raise ValueError(f"Incomplete window: {trace}/{window}")
                    for spacing in (F(1), F(1, 4)):
                        data = build(rows, 256, fixture, model, spacing, "split", cap)
                        releases = ("timer",) if args.timer_ms else ("start", "finish")
                        for rate, budget, release, host in product((1, 32), (128, 384), releases, (0, 128)):
                            cost = dict(fixture["cost"], recompute=grid)
                            for path, row in io.items():
                                scale = 1 if path == "device_copy" else rate
                                cost[path] = dict(startup_ms=str(F(row["startup_ms"]) * scale),
                                                  bytes_per_ms=str(F(row["bytes_per_ms"]) / scale))
                            exp = Experiment(dict(data, cost=cost, overlap_slowdown=overlap,
                                                  staged_checkpoint_cache_mib=host,
                                                  scheduling=dict(data["scheduling"], prefetch_release=release,
                                                                  prefetch_hold_ms=args.timer_ms or 0)),
                                             model, budget)
                            evaluate(exp, dict(cohort=args.cohort, trace=trace, window=window, spacing=float(spacing),
                                               budget_mib=budget, io_scale=rate, release=release, host_mib=host),
                                     writer, not args.zero_only)
                    out.flush()


if __name__ == "__main__":
    main()
