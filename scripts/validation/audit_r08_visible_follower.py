"""Measure online-visible single-follower lookahead on R08 development witnesses."""

import argparse
from collections import Counter
from fractions import Fraction
import gzip
import inspect
import json
from statistics import median
from textwrap import dedent

from scripts.cpu.replay.study_r08_capacity import CONFIG, ROOT, cases, open_exp, read, response, run
from scripts.cpu.replay.study_r07_arrivals import Experiment, score
from scripts.cpu.replay.study_bailian import p95


def diagnostic_executor():
    source = dedent(inspect.getsource(Experiment._run))
    signature = "pause_at_head=None, early_portfolio=False"
    decision = "            fastest = min(scores, key=lambda c: (scores[c], -c))\n"
    assert source.count(signature) == source.count(decision) == 1
    source = source.replace(signature, signature + ", follower_audit=None, include_follower=True, "
                            "follower_extra=None, ssd_gate=False, sparse_follower=False, "
                            "require_disagreement=False, reuse_scores=False, reuse_prefix=False, "
                            "score_audit=None, prefix_capture=None, capture_inline=False, gap_audit=None, "
                            "short_margin_max=None, near_follower=False, two_current=False, "
                            "ssd_compute_min_pct=None")
    # Diagnostic forecasts can checkpoint the common prefix of alternatives.
    pause_restriction = 'or not lean_rollout or self.dynamic'
    pause_return = '            return {"completion_ms": completions, "events_processed": events_processed,\n                    "snapshot": state}'
    assert source.count(pause_restriction) == source.count(pause_return) == 1
    source = source.replace(pause_restriction, 'or not lean_rollout')
    pause_gate = "if pause_at_head is not None and advanced and head == pause_at_head:"
    assert source.count(pause_gate) == 1
    source = source.replace(pause_gate, """if (pause_at_head is not None and advanced and head == pause_at_head
                or prefix_capture is not None and advanced and head == prefix_capture[0]):""")
    source = source.replace(pause_return, """            if self.dynamic:
                state += (copy_until, compute_factor, transfer_factor)
            if prefix_capture is not None:
                prefix_capture.append({"completion_ms": list(completions), "snapshot": state})
            else:
""" + "\n".join("    " + line for line in pause_return.splitlines()))
    forecast_signature = "cutoff=None):\n                nonlocal decision_replays"
    assert source.count(forecast_signature) == 1
    source = source.replace(forecast_signature,
                            "cutoff=None, prefix_stop=None, prefix_replay=None, capture=None):\n                nonlocal decision_replays")
    vanilla_run = '                run = self._run("fixed", 0, (*selected[:i], *plan), max_combinations, n_visible,'
    assert source.count(vanilla_run) == 1
    source = source.replace(vanilla_run, """                prior = []
                if prefix_replay is not None:
                    frozen = prefix_replay["snapshot"]
                    state = (*frozen[:3], dict(frozen[3]), set(frozen[4]), *frozen[5:8],
                             set(frozen[8]), frozen[9], dict(frozen[10]), *frozen[11:])
                    prior = prefix_replay["completion_ms"]
                runner = (lambda *a, **kw: _run(self, *a, pause_at_head=prefix_stop, prefix_capture=capture, **kw)
                          ) if prefix_stop is not None or capture is not None else self._run
                run = runner("fixed", 0, (*selected[:i], *plan), max_combinations, n_visible,""")
    forecast_returns = '                return run["completion_ms"]\n'
    assert source.count(forecast_returns) == 1
    source = source.replace(forecast_returns, """                if prefix_stop is not None:
                    return run
                return prior + run["completion_ms"]
""")
    source = source.replace(decision, decision + """
            if (follower_audit is not None and i + horizon + 1 < arrived
                    and (not require_disagreement or fastest != fallback)
                    and (short_margin_max is None or
                         scores[fallback] - scores[fastest] <= short_margin_max * (visible - head))):
                short_view, short_tail, short_scores = visible, later, scores
                short_choice = fastest
                visible = arrived
                follower = next((j for j in range(i + 1, visible) if len(self.eligible[j]) > 1), None)
                def needs_ssd(request):
                    return any(c and self.objects[deps[-1]]["tier"] == "SSD" and deps[-1] not in ready
                               for c, deps in self.candidates[request].items())
                near = (follower is not None and follower - i <= self.window
                        and observable[short_choice]["prefetch_fits"])
                row = observable[short_choice]
                costly_ssd = (ssd_compute_min_pct is None or
                              100 * row["incremental_ssd_ms"] >= ssd_compute_min_pct * row["serial_compute_ms"])
                if (costly_ssd and (not near_follower or near)
                        and (not ssd_gate or (follower is not None and needs_ssd(i) and needs_ssd(follower)))):
                    if follower_extra is not None:
                        visible = min(visible, max(i + horizon + 1, (follower or i) + follower_extra + 1))
                    if not include_follower:
                        follower = None
                    if reuse_scores and not self.host_cache_bytes:
                        for j in range(i + 1, visible):
                            if j not in serial_choices:
                                serial_choices[j] = serial_choice(j)
                        later = [serial_choices[j] for j in range(i + 1, visible)]
                    else:
                        later = [serial_choice(j) for j in range(i + 1, visible)]
                    scores = {}
                    current = tuple(sorted({short_choice, fallback})) if two_current else choices
                    for boundary in current:
                        options = self.eligible[follower] if follower is not None else (None,)
                        if sparse_follower and follower is not None:
                            options = tuple(sorted({later[follower - i - 1], max(options)}))
                        prefix = capture = None
                        if (reuse_prefix and follower is not None and len(options) > 1
                                and follower - self.window > head):
                            stop = follower - self.window
                            if capture_inline:
                                capture = [stop]
                            else:
                                prefix = forecast((boundary, *later), visible, prefix_stop=stop)
                                assert "snapshot" in prefix
                        for alternative in options:
                            tail = list(later)
                            if follower is not None:
                                tail[follower - i - 1] = alternative
                            if (reuse_scores and visible == short_view and boundary in short_scores
                                    and tuple(tail) == short_tail):
                                value = short_scores[boundary]
                            else:
                                value = sum(forecast((boundary, *tail), visible, prefix_replay=prefix, capture=capture))
                                if capture is not None:
                                    _, prefix = capture
                                    capture = None
                            scores[boundary] = min(scores.get(boundary, value), value)
                    if score_audit is not None:
                        score_audit.append((i, visible, tuple(sorted(scores.items()))))
                    revised = min(scores, key=lambda c: (scores[c], -c))
                    fastest = fallback if scores[fallback] == scores[revised] else revised
                    follower_audit.append((i, visible, follower, fastest))
                    if gap_audit is not None:
                        gap_audit.append(dict(index=i, frozen=short_choice, revised=fastest,
                                              fallback=fallback, follower=follower,
                                              head_distance=i - head,
                                              follower_distance=follower - i if follower is not None else None,
                                              ssd_ms=float(observable[short_choice]["incremental_ssd_ms"]),
                                              compute_ms=float(observable[short_choice]["serial_compute_ms"]),
                                              queue_debt_ms=float(observable[short_choice]["queue_debt_ms"]),
                                              fits=observable[short_choice]["prefetch_fits"],
                                              short_gain_ms=float(short_scores[fallback] - short_scores[short_choice]),
                                              short_mean_gain_ms=float((short_scores[fallback] - short_scores[short_choice])
                                                                       / (short_view - head))))
""")
    namespace = dict(Experiment._run.__globals__)
    exec(compile(source, "<r08-visible-follower-diagnostic>", "exec"), namespace)
    return namespace["_run"]


def audit(exp, row, executor, repeats, follower_extra, sweep, ssd_gate, sparse_follower, match_executor, require_disagreement, pair_triggers, reuse_scores, reuse_prefix, capture_inline, short_margin_max, near_follower, two_current, ssd_compute_min_pct):
    base = run(exp, "local-cost")
    frozen = run(exp, "entry-replay")
    assert frozen["boundaries_tokens"] == row["plan"]
    assert frozen["response_mean_ms"] == Fraction(row["main_exact"])

    def execute(log, *, timed=False, include_follower=True, extra=follower_extra, disagreement=require_disagreement, reuse=reuse_scores, prefix=reuse_prefix, audit_scores=None, inline=capture_inline, gaps=None, margin=short_margin_max, near=near_follower, shortlist=two_current, io_ratio=ssd_compute_min_pct):
        fast = match_executor and exp._stationary_costs
        variant = "entry" if match_executor else "entry-replay"
        return executor(exp, "snapshot-screened", 0, None, 10000, len(exp.requests),
                        score_candidate=score, screen_mode="unsettled", queue_horizon=4,
                        screen_search=variant, cached_costs=fast, committed_choices=fast, scalar_replay=fast,
                        follower_audit=log, include_follower=include_follower, follower_extra=extra, ssd_gate=ssd_gate, sparse_follower=sparse_follower, require_disagreement=disagreement, reuse_scores=reuse, reuse_prefix=prefix, score_audit=audit_scores, capture_inline=inline, gap_audit=gaps, short_margin_max=margin, near_follower=near, two_current=shortlist, ssd_compute_min_pct=io_ratio, inject_measured=timed)

    control = execute(None)
    for field in ("boundaries_tokens", "completion_ms", "transfers"):
        assert control[field] == frozen[field], field
    events = []
    exact_scores = [] if reuse_prefix else None
    gap_audit = []
    candidate = execute(events, audit_scores=exact_scores, gaps=gap_audit)

    reference_replays = reference_events_processed = None
    if reuse_scores or reuse_prefix:
        reference_events = []
        reference_scores = [] if reuse_prefix else None
        reference = execute(reference_events, reuse=False, prefix=False, audit_scores=reference_scores)
        assert reference_scores == exact_scores
        assert reference_events == events
        for field in ("boundaries_tokens", "completion_ms", "transfers", "prefetch_reserved_at_ms"):
            assert reference[field] == candidate[field], field
        reference_replays = reference["decision_replays"]
        reference_events_processed = reference["decision_events"]

    def compare(baseline, incumbent, result):
        losses = [x-y for x,y in zip(result["completion_ms"],baseline["completion_ms"],strict=True)]
        gain = incumbent["response_mean_ms"] - result["response_mean_ms"]
        reference_saving = baseline["response_mean_ms"] - incumbent["response_mean_ms"]
        return dict(gain_over_frozen_ms=float(gain),
                    p95_loss_vs_frozen_ms=float(p95(response(exp, result)) - p95(response(exp, incumbent))),
                    retained_saving=float((reference_saving + gain) / reference_saving) if reference_saving > 0 else None,
                    decision_ms=sum(t for _,t in result.get("decision_profile_ns", []))/1e6,
                    frozen_decision_ms=sum(t for _,t in incumbent.get("decision_profile_ns", []))/1e6,
                    worst_loss_vs_local_ms=float(max(losses)), late_vs_local=sum(x > 10 for x in losses),
                    late_indices=[i for i, x in enumerate(losses) if x > 10])

    sensitivity = []
    if sweep:
        for extra in range(17):
            log = []
            result = execute(log, extra=extra)
            sensitivity.append(dict(extra=extra, max_visible=max((n-i-1 for i,n,_,_ in log), default=0),
                                    changed=[i for i,(a,b) in enumerate(zip(row["plan"], result["boundaries_tokens"],
                                                                           strict=True)) if a != b],
                                    **compare(base, frozen, result)))
    timing = []
    direct_pairs = []
    for repeat in range(repeats):
        actions = {"base": lambda: run(exp, "local-cost", timed=True),
                   "frozen": lambda: run(exp, "entry-replay", timed=True),
                   "candidate": lambda: execute([], timed=True)}
        if pair_triggers:
            if ssd_compute_min_pct is not None:
                actions["comparison"] = lambda: execute([], timed=True, io_ratio=None)
            elif two_current:
                actions["comparison"] = lambda: execute([], timed=True, shortlist=False)
            elif near_follower:
                actions["comparison"] = lambda: execute([], timed=True, near=False)
            elif short_margin_max is not None:
                actions["comparison"] = lambda: execute([], timed=True, margin=None)
            elif capture_inline:
                actions["comparison"] = lambda: execute([], timed=True, inline=False)
            elif reuse_scores or reuse_prefix:
                actions["comparison"] = lambda: execute([], timed=True, reuse=False, prefix=False)
            else:
                actions["comparison"] = lambda: execute([], timed=True, disagreement=False)
        order = tuple(actions)
        rotated = order[repeat % 3:] + order[:repeat % 3]
        results = {name: actions[name]() for name in (rotated if repeat < 3 else rotated[::-1])}
        metrics = compare(results["base"], results["frozen"], results["candidate"])
        reference = compare(results["base"], results["frozen"], results["frozen"])
        metrics["frozen_late_vs_local"] = reference["late_vs_local"]
        metrics["frozen_late_indices"] = reference["late_indices"]
        timing.append(metrics)
        if pair_triggers:
            reference = compare(results["base"], results["frozen"], results["comparison"])
            direct_pairs.append(dict(
                cpu_saved_ms=reference["decision_ms"] - metrics["decision_ms"],
                mean_saved_ms=metrics["gain_over_frozen_ms"] - reference["gain_over_frozen_ms"],
                p95_saved_ms=reference["p95_loss_vs_frozen_ms"] - metrics["p95_loss_vs_frozen_ms"],
                comparison=reference,
            ))
    candidate_late = Counter(i for t in timing for i in t["late_indices"])
    frozen_late = Counter(i for t in timing for i in t["frozen_late_indices"])
    added_persistent = sorted(i for i, n in candidate_late.items() if n >= 4 and frozen_late[i] < 4)
    return dict(key=row["key"], events=events, short_gaps=gap_audit, added_persistent=added_persistent,
                changed=[i for i, (a,b) in enumerate(zip(control["boundaries_tokens"],
                                                         candidate["boundaries_tokens"])) if a != b],
                replays=candidate["decision_replays"], reference_replays=reference_replays,
                replay_events=candidate["decision_events"], reference_events_processed=reference_events_processed, zero=compare(base, frozen, candidate), sensitivity=sensitivity, direct_pairs=direct_pairs,
                horizon_only=compare(base, frozen, execute([], include_follower=False)), timing=timing,
                timed_gain_median_ms=median(t["gain_over_frozen_ms"] for t in timing) if timing else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    datasets = parser.add_mutually_exclusive_group()
    datasets.add_argument("--public-controls", action="store_true")
    datasets.add_argument("--public-matrix", action="store_true")
    datasets.add_argument("--public-design", action="store_true")
    datasets.add_argument("--public-native", action="store_true")
    parser.add_argument("--time-controls", action="store_true", help="Inject measured decision time on public controls")
    views = parser.add_mutually_exclusive_group()
    views.add_argument("--bounded-view", action="store_true")
    views.add_argument("--view-extra", type=int, choices=range(17), metavar="0..16")
    parser.add_argument("--sweep-view", action="store_true")
    parser.add_argument("--ssd-gate", action="store_true")
    parser.add_argument("--sparse-follower", action="store_true")
    parser.add_argument("--match-executor", action="store_true")
    parser.add_argument("--require-disagreement", action="store_true")
    parser.add_argument("--pair-triggers", action="store_true")
    parser.add_argument("--reuse-scores", action="store_true")
    parser.add_argument("--reuse-prefix", action="store_true")
    parser.add_argument("--capture-inline", action="store_true")
    parser.add_argument("--short-margin-max", type=int)
    parser.add_argument("--near-follower", action="store_true")
    parser.add_argument("--two-current", action="store_true")
    parser.add_argument("--ssd-compute-min-pct", type=int)
    args = parser.parse_args()
    if args.ssd_compute_min_pct is not None and args.ssd_compute_min_pct <= 0:
        parser.error("--ssd-compute-min-pct must be positive")
    if args.short_margin_max is not None and args.short_margin_max < 0:
        parser.error("--short-margin-max must be nonnegative")
    if args.capture_inline and not args.reuse_prefix:
        parser.error("--capture-inline requires --reuse-prefix")
    if args.pair_triggers and not (args.require_disagreement and args.ssd_gate):
        parser.error("--pair-triggers requires --require-disagreement and --ssd-gate")
    if args.time_controls and not (args.public_controls or args.public_matrix or args.public_design or args.public_native):
        parser.error("--time-controls requires a public dataset")
    cfg = read(CONFIG)
    names = ["r08-holdout.jsonl.gz"]
    split = "design" if args.public_design else "holdout"
    if args.public_native:
        names = ["r08-native.jsonl.gz"]
        selected = set()
    elif args.public_design:
        names = ["r08-design.jsonl.gz"]
        selected = set()
    elif args.public_controls or args.public_matrix:
        cfg["holdout"]["qwen_coder_blksz_16.jsonl"] = [97, 98, 99, 101]
        names.append("r08-replacement.jsonl.gz")
        selected = {(trace, window, "27b", 16, "512")
                    for trace, windows in cfg["holdout"].items() for window in windows}
    else:
        trace = "qwen_traceB_blksz_16.jsonl"
        cfg["holdout"] = {trace: [481, 482]}
        selected = {(trace, 481, "4b", 12, "176"), (trace, 482, "27b", 16, "384")}

    def matches(key):
        if args.public_native:
            return True
        if args.public_matrix or args.public_design:
            return key["window"] in cfg[split].get(key["trace"], ())
        return (key["section"] == "fixed" and
                (key["trace"], key["window"], key["model"], key["rate"], key["level"]) in selected)

    archived = {}
    for name in names:
        with gzip.open(ROOT / "docs/evidence" / name, "rt", encoding="utf-8") as source:
            for row in map(json.loads, source):
                if matches(row["key"]):
                    key = tuple(sorted(row["key"].items()))
                    assert key not in archived
                    archived[key] = row
    expected = 204 if args.public_native else 864 if args.public_matrix else 144 if args.public_design else len(selected)
    assert len(archived) == expected
    executor = diagnostic_executor()

    def experiments():
        if args.public_native:
            from scripts.cpu.replay.study_r08_native import CFG, get_exp, make_costs, tasks

            def evidence(name):
                with gzip.open(ROOT / "docs/evidence" / name, "rt", encoding="utf-8") as source:
                    return json.load(source)

            native = read(CFG)
            costs = make_costs(native, evidence("r08-native-prefill.json.gz"), evidence("r08-native-io.json.gz"))
            for key, data, model in tasks(native):
                yield key, get_exp(key, data, model, costs)
        else:
            for key, data, model, scale in cases(cfg, split):
                if matches(key):
                    yield key, open_exp(data, model, key["rate"], key["capacity_bytes"], scale)

    timed = args.time_controls or not (args.public_controls or args.public_matrix or args.public_design or args.public_native)
    for key, exp in experiments():
        print(json.dumps(audit(exp, archived.pop(tuple(sorted(key.items()))), executor,
                               repeats=6 if timed else 0,
                               follower_extra=4 if args.bounded_view else args.view_extra,
                               sweep=args.sweep_view, ssd_gate=args.ssd_gate,
                               sparse_follower=args.sparse_follower, match_executor=args.match_executor,
                               require_disagreement=args.require_disagreement, pair_triggers=args.pair_triggers,
                               reuse_scores=args.reuse_scores, reuse_prefix=args.reuse_prefix,
                               capture_inline=args.capture_inline, short_margin_max=args.short_margin_max,
                               near_follower=args.near_follower, two_current=args.two_current,
                               ssd_compute_min_pct=args.ssd_compute_min_pct)))
    assert not archived


if __name__ == "__main__":
    main()
