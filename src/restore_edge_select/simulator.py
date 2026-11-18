"""Phase 1 deterministic event simulation. All clock arithmetic uses fractions."""

from bisect import bisect_right
from collections import Counter
from fractions import Fraction
from itertools import product
from math import prod
from time import perf_counter_ns

from .adaptive import AdaptiveConfig, select as adaptive_select
from .prefill_cost import build_prefill_cost


PAIR_POLICIES = {"two-request", "two-request-capacity", "two-request-transfer", "two-request-either",
                 "two-request-shortlist", "two-request-io-floor", "two-request-backlog",
                 "two-request-backlog-io", "two-request-checkpoint-floor", "two-request-exposed-io",
                 "two-request-exposed-floor", "two-request-stable-io", "two-request-robust"}
PROBE_POLICIES = {"two-request-io-floor", "two-request-backlog", "two-request-backlog-io",
                  "two-request-checkpoint-floor", "two-request-exposed-io", "two-request-exposed-floor",
                  "two-request-stable-io", "two-request-robust"}
IO_PATHS = ("ram_to_device", "ssd_to_device")


class OfflineBranch(Exception):
    """A fixed replay reached a request with an unassigned boundary."""


def prefix(runs, length):
    result = []
    for run in runs:
        if length <= 0:
            break
        count = min(length, run["count"])
        token = run["token_id"]
        if result and result[-1][0] == token:
            result[-1] = (token, result[-1][1] + count)
        else:
            result.append((token, count))
        length -= count
    return tuple(result)


def number(value):
    return Fraction(str(value))


class Experiment:
    def __init__(self, trace, spec, prefetch_mib):
        self.trace = trace
        self.objects = {obj["id"]: obj for obj in trace["objects"]}
        if len(self.objects) != len(trace["objects"]):
            raise ValueError("Duplicate object ID")
        self.requests = sorted(trace["requests"], key=lambda r: r["arrival_ms"])
        self.lengths = [sum(run["count"] for run in r["input_token_runs"]) for r in self.requests]
        self.arrivals = [number(r["arrival_ms"]) for r in self.requests]
        self.candidates = [
            {c["boundary_tokens"]: tuple(c["load_order"]) for c in r["candidates"]}
            for r in self.requests
        ]
        if len({r["id"] for r in self.requests}) != len(self.requests):
            raise ValueError("Duplicate request ID")
        memory = trace["memory"]
        self.alignment = memory["extra_alignment_bytes"]
        self.initial = set(trace["scheduling"]["initial_device_object_ids"])
        self.window = trace["scheduling"]["future_window_requests"]
        self.release_at = trace["scheduling"].get("prefetch_release", "start")
        self.release_delay = number(trace["scheduling"].get("prefetch_hold_ms", 0))
        if self.release_at not in {"start", "finish", "timer"} or (
                (self.release_delay > 0) != (self.release_at == "timer")) or self.release_delay < 0:
            raise ValueError("Prefetch release uses start, finish, or a positive service-start timer")
        self.cost = trace["cost"]
        topology = trace.get("transfer_topology", "shared")
        if topology not in {"shared", "per-tier"}:
            raise ValueError("Unsupported transfer topology")
        self.per_tier = topology == "per-tier"
        coupling = trace.get("lane_slowdown")
        self.lane_slowdown = {}
        if coupling is not None:
            if not self.per_tier or set(coupling) != set(IO_PATHS):
                raise ValueError("Lane slowdown requires both per-tier lanes")
            self.lane_slowdown = {path: number(value) for path, value in coupling.items()}
            if min(self.lane_slowdown.values()) < 1:
                raise ValueError("Lane slowdown factors must be at least one")
        self.overlap = {}
        for path, rates in trace.get("overlap_slowdown", {}).items():
            if path not in IO_PATHS or set(rates) != {"compute", "transfer"}:
                raise ValueError("Expected compute and transfer factors for a known I/O path")
            compute, transfer = (number(rates[k]) for k in ("compute", "transfer"))
            if min(compute, transfer) < 1:
                raise ValueError("Overlap slowdown factors must be at least one")
            if (compute, transfer) != (1, 1):
                self.overlap[path] = (compute, transfer)
        self.joint_prefill = {}
        if "joint_prefill_slowdown" in trace:
            rates = trace["joint_prefill_slowdown"]
            if not self.lane_slowdown or set(rates) != {"compute", *IO_PATHS}:
                raise ValueError("Joint prefill rates require a coupled two-lane model")
            self.joint_prefill = {path: number(value) for path, value in rates.items()}
            if min(self.joint_prefill.values()) < 1:
                raise ValueError("Joint prefill slowdown factors must be at least one")
        if self.per_tier and self.overlap and not self.joint_prefill:
            raise ValueError("Use explicit joint prefill rates for compute contention on two lanes")
        self.dynamic = bool(self.overlap or self.joint_prefill)
        self.host_cache_bytes = number(trace.get("staged_checkpoint_cache_mib", 0)) * trace["units"]["bytes_per_mib"]
        if self.host_cache_bytes < 0 or (self.host_cache_bytes and self.per_tier):
            raise ValueError("Staged checkpoint retention requires nonnegative capacity and shared I/O")
        self._path_times = {}
        path_epochs = trace.get("io_path_speed_epochs", {})
        if set(path_epochs) - set(IO_PATHS):
            raise ValueError("Unknown I/O path in speed epochs")
        self.io_epochs = {}
        for path in IO_PATHS:
            epochs = [(number(t), number(speed)) for t, speed in
                      path_epochs.get(path, trace.get("io_speed_epochs", [(0, 1)]))]
            if (not epochs or epochs[0][0] != 0 or any(a[0] >= b[0] for a, b in zip(epochs, epochs[1:]))
                    or min(speed for _, speed in epochs) <= 0):
                raise ValueError("I/O speed epochs require increasing times from zero and positive speeds")
            self.io_epochs[path] = tuple(zip(*epochs))
        self._prefill_cost = build_prefill_cost(self.cost["recompute"], spec)
        mib = trace["units"]["bytes_per_mib"]
        self.budget = prefetch_mib * mib
        self.fixed = memory["fixed_mib"] * mib + self.size(self.initial)
        self.execution = memory["execution_mib"] * mib
        self.checkpoint = spec["payload"]["checkpoint_bytes"]
        kv_bytes = spec["payload"]["attention_bytes_per_token"]
        block = memory["kv_block_tokens"]
        if (block <= 0 or min(self.lengths, default=0) <= 0 or self.alignment < 0
                or max(self.lengths, default=0) > spec["context_tokens"]
                or any(run["count"] <= 0 for r in self.requests for run in r["input_token_runs"])):
            raise ValueError("Invalid token count, block size or alignment")
        self.run_bytes = [
            length * spec["payload"]["attention_bytes_per_token"]
            + self.checkpoint + memory["workspace_mib"] * mib
            + ((length + memory["kv_block_tokens"] - 1) // memory["kv_block_tokens"] + 1)
            * self.alignment for length in self.lengths
        ]
        if not self.requests or self.window < 0 or min(self.arrivals) < 0:
            raise ValueError("Expected a nonempty trace, nonnegative arrivals and window")
        if self.budget < 0 or self.fixed + self.execution + self.budget > memory["device_mib"] * mib:
            raise ValueError("Memory budgets exceed device capacity or prefetch budget is negative")
        if max(self.run_bytes) > self.execution:
            raise ValueError("Execution budget cannot cover full recomputation")
        self.eligible = []
        mentions = Counter(obj for choices in self.candidates
                           for obj in {deps[-1] for deps in choices.values() if deps})
        for i, choices in enumerate(self.candidates):
            if len(choices) != len(self.requests[i]["candidates"]):
                raise ValueError("Duplicate candidate boundary")
            if choices.get(0) != ():
                raise ValueError("Each request needs an empty boundary zero")
            allowed = []
            for c, deps in sorted(choices.items()):
                if (not 0 <= c <= self.lengths[i] or (c > 0 and not deps)
                        or (c % block and c != self.lengths[i])):
                    raise ValueError("Invalid boundary")
                if c:
                    if len(deps) != (c + block - 1) // block + 1:
                        raise ValueError(f"Wrong dependency count at boundary {c}")
                    runs = self.requests[i]["input_token_runs"]
                    for j, obj in enumerate(deps):
                        item = self.objects[obj]
                        if j == len(deps) - 1:
                            end = c
                            valid = (item["kind"] == "recurrent_and_conv_checkpoint"
                                     and item["boundary_tokens"] == c
                                     and item["bytes"] == self.checkpoint)
                        else:
                            start, end = j * block, min((j + 1) * block, c)
                            valid = (item["kind"] == "attention_kv"
                                     and item["start_token_exclusive"] == start
                                     and item["end_token_inclusive"] == end
                                     and item["bytes"] == (end - start) * kv_bytes)
                        if not valid or prefix(item["prefix_token_runs"], end) != prefix(runs, end):
                            raise ValueError(f"Invalid dependency {obj} at boundary {c}")
                        if obj not in self.initial:
                            self.transfer_time(obj)
                shared = c and (deps[-1] in self.initial or mentions[deps[-1]] > 1)
                # A shared source remains charged to its existing pool.
                # The private checkpoint is already included in run_bytes.
                if c < self.lengths[i] and shared:
                    if "device_copy" not in self.cost:
                        raise ValueError("Shared checkpoint continuation requires cost.device_copy")
                    self.path_time("device_copy", self.objects[deps[-1]]["bytes"])
                allowed.append(c)
            self.eligible.append(allowed)
        self._suffix_times = [{c: self.recompute(i, c) for c in choices}
                              for i, choices in enumerate(self.eligible)]
        self._transfer_times = {obj: self.transfer_time(obj) for obj in self.objects if obj not in self.initial}
        self._retained_times = {
            obj: self.path_time("ram_to_device", item["bytes"]) for obj, item in self.objects.items()
            if self.host_cache_bytes and obj not in self.initial and item["tier"] == "SSD"
            and item["kind"] == "recurrent_and_conv_checkpoint" and item["bytes"] <= self.host_cache_bytes
        }
        self._actual_transfer_times = self._transfer_times | {
            obj: number(value) for obj, value in trace.get("actual_transfer_ms_by_object", {}).items()
        }
        if (self._actual_transfer_times.keys() != self._transfer_times.keys()
                or any(value <= 0 for value in self._actual_transfer_times.values())):
            raise ValueError("Actual transfer times require positive costs for nonresident objects")
        if self.dynamic and (trace.get("actual_transfer_ms_by_object") or
                             any(len(t) != 1 for t, _ in self.io_epochs.values())):
            raise ValueError("Overlap model currently requires stationary known I/O")

    def size(self, ids):
        return sum(self.objects[obj]["bytes"] + self.alignment for obj in ids)

    def path_time(self, path, size):
        key = (path, size)
        if key in self._path_times:
            return self._path_times[key]
        p = self.cost[path]
        startup, bandwidth = number(p["startup_ms"]), number(p["bytes_per_ms"])
        if startup < 0 or bandwidth <= 0:
            raise ValueError(f"Invalid transfer cost: {path}")
        value = startup + size / bandwidth
        self._path_times[key] = value
        return value

    def transfer_time(self, obj):
        item = self.objects[obj]
        return self.path_time({"RAM": "ram_to_device", "SSD": "ssd_to_device"}[item["tier"]],
                              item["bytes"])

    def recompute(self, i, c):
        length = self.lengths[i]
        return self._prefill_cost(c, length - c)

    def run(self, policy="local-cost", *, threshold_tokens=0, boundaries=None,
            max_combinations=10000, estimate=None, guard_ms=0, transfer_trigger_ms=0,
            screen_ms=0, pair_ms=0, probe_ms=0, telemetry_lag_ms=0, stable_ms=0,
            profile_decisions=False, inject_measured=False, observe_candidates=None, score_candidate=None,
            intervention_budget_mib=None, decision_overrides=None, screen_mode=None, min_gain_ms=0,
            future_cost_cases=None, queue_horizon=None, screen_selector=None, screen_observer=None,
            screen_search="full", adaptive=None):
        if policy == "adaptive":
            adaptive = AdaptiveConfig() if adaptive is None else adaptive
            if (not isinstance(adaptive, AdaptiveConfig) or self.per_tier or self.host_cache_bytes
                    or future_cost_cases or screen_search != "full" or queue_horizon is not None
                    or (adaptive.audit and (profile_decisions or inject_measured))):
                raise ValueError("Adaptive routing requires the single-lane model and its own search configuration")
        elif adaptive is not None:
            raise ValueError("Adaptive configuration requires the adaptive policy")
        estimate_modes = {"estimate", "estimate-cross-tier", "estimate-cross-tier-guard",
                          "estimate-prefetch-fallback"}
        if screen_search not in {"full", "lean", "compute", "cutoff", "entry"} | estimate_modes or (
                screen_search != "full" and policy != "snapshot-screened"):
            raise ValueError("Search variants require a screened queue policy")
        if (screen_mode == "classified") != (screen_selector is not None):
            raise ValueError("Classified screening requires a pre-decision selector")
        if screen_observer is not None and (screen_selector is None or inject_measured or profile_decisions):
            raise ValueError("Screen observation requires uninstrumented classified screening")
        if queue_horizon is not None and (policy != "snapshot-screened" or
                                          type(queue_horizon) is not int or queue_horizon < 0):
            raise ValueError("Queue horizon requires screening and a nonnegative integer (zero means all arrived)")
        if screen_search in estimate_modes and (
                screen_selector is None or screen_observer is not None or future_cost_cases):
            raise ValueError("Estimate selection requires a classifier and no forecast observers or probes")
        if observe_candidates is not None and (policy != "remaining-wait-live" or inject_measured or profile_decisions):
            raise ValueError("Candidate observation requires uninstrumented remaining-wait-live timing")
        if future_cost_cases and ((observe_candidates is None and policy != "snapshot-screened")
                                  or "nominal" in future_cost_cases):
            raise ValueError("Future-cost probes require observation or screening and non-nominal names")
        if future_cost_cases:
            future_cost_cases = dict(future_cost_cases)
            for name, case in future_cost_cases.items():
                if len(case) == 2:
                    continue
                if len(case) != 3 or not self.per_tier or set(case[2]) != set(IO_PATHS):
                    raise ValueError("Lane slowdown forecast requires both per-tier lanes")
                factors = {path: number(value) for path, value in case[2].items()}
                if min(factors.values()) < 1:
                    raise ValueError("Lane slowdown factors must be at least one")
                future_cost_cases[name] = (*case[:2], factors)
        if (policy in {"snapshot-score", "snapshot-screened"}) != (score_candidate is not None):
            raise ValueError("Snapshot policies require a pre-decision scoring function")
        min_gain = number(min_gain_ms)
        if ((policy == "snapshot-screened") != (screen_mode is not None)
                or screen_mode not in (None, "disagree", "unsettled", "classified") or min_gain < 0
                or (min_gain and screen_mode is None)):
            raise ValueError("Expected valid screening mode and nonnegative gain threshold")
        overrides = dict(decision_overrides or {})
        if overrides and (policy not in {"local-cost", "snapshot-score", "remaining-wait-live", "snapshot-screened"}
                          or inject_measured or profile_decisions):
            raise ValueError("Decision overrides require a supported zero-overhead policy")
        if any(type(i) is not int or not 0 <= i < len(self.eligible) or c not in self.eligible[i]
               for i, c in overrides.items()):
            raise ValueError("Decision override must select an eligible boundary")
        intervention_budget = None
        if intervention_budget_mib is not None:
            mib = self.trace["units"]["bytes_per_mib"]
            intervention_budget = number(intervention_budget_mib) * mib
            if (observe_candidates is None or intervention_budget <= self.budget or
                    self.fixed + self.execution + intervention_budget > self.trace["memory"]["device_mib"] * mib):
                raise ValueError("Intervention requires candidate observation and a feasible larger prefetch budget")
        estimate = {k: number(v) for k, v in (estimate or {}).items()}
        transfer_trigger = number(transfer_trigger_ms)
        screen, pair_cost = number(screen_ms), number(pair_ms)
        probe_cost = number(probe_ms)
        lag, stable = number(telemetry_lag_ms), number(stable_ms)
        if self.dynamic and (estimate or lag or policy in PAIR_POLICIES):
            raise ValueError("Overlap model requires known work rates and single-request forecasts")
        if self.per_tier and (policy not in {"fixed", "local-cost", "remaining-wait-live", "snapshot-screened"}
                              or estimate or lag or self._actual_transfer_times != self._transfer_times
                              or any(len(t) != 1 for t, _ in self.io_epochs.values())):
            raise ValueError("Per-tier contrast requires stationary known costs and an exact supported policy")
        if self.host_cache_bytes and (policy not in {"fixed", "local-cost", "remaining-wait-live", "snapshot-screened"}
                                     or estimate or lag or self._actual_transfer_times != self._transfer_times
                                     or any(len(t) != 1 or s != (1,) for t, s in self.io_epochs.values())):
            raise ValueError("Staged checkpoint retention requires stationary known costs and a supported policy")
        if self.release_at != "start" and policy not in {"fixed", "local-cost", "remaining-wait-live",
                                                        "snapshot-screened", "adaptive"}:
            raise ValueError("Delayed prefetch release requires a supported policy")
        if set(estimate) - {"ram_to_device", "ssd_to_device", "device_copy", "compute"} or any(
                v <= 0 for v in estimate.values()) or min(number(guard_ms), transfer_trigger, screen, pair_cost,
                                                          probe_cost, lag, stable) < 0:
            raise ValueError("Invalid duration multiplier or negative threshold")
        return self._run(policy, threshold_tokens, boundaries, max_combinations, len(self.requests),
                         estimate=estimate, guard=number(guard_ms), transfer_trigger=transfer_trigger,
                         screen=screen, pair_cost=pair_cost, probe_cost=probe_cost, lag=lag, stable=stable,
                         profile_decisions=profile_decisions, inject_measured=inject_measured,
                         observe_candidates=observe_candidates, score_candidate=score_candidate,
                         intervention_budget=intervention_budget, decision_overrides=overrides,
                         screen_mode=screen_mode, min_gain=min_gain, future_cost_cases=future_cost_cases,
                         queue_horizon=queue_horizon, screen_selector=screen_selector,
                         screen_observer=screen_observer, screen_search=screen_search, adaptive=adaptive)

    def _run(self, policy, threshold_tokens, boundaries, max_combinations, n, *,
             snapshot=None, estimate=None, guard=0, transfer_trigger=0, screen=0, pair_cost=0, probe_cost=0,
             lag=0, stable=0, forecast_speed=None, profile_decisions=False, inject_measured=False,
             observe_candidates=None, score_candidate=None, intervention_budget=None,
             prefetch_budget=None, active_kv_handoff=True, decision_overrides=None,
             screen_mode=None, min_gain=0, future_cost_cases=None, future_cost=None, queue_horizon=None,
             screen_selector=None, screen_observer=None, screen_search="full",
             completion_cutoff=None, lean_rollout=False, adaptive=None, offline_probe=None,
             pause_at_head=None):
        if pause_at_head is not None and (
                snapshot is None or policy != "fixed" or not lean_rollout or self.dynamic
                or self.per_tier or self.host_cache_bytes or self.release_at != "start"
                or completion_cutoff is not None or future_cost is not None
                or not snapshot[1] < pause_at_head <= n or snapshot[2] != n
                or self.arrivals[n-1] > snapshot[0]):
            raise ValueError("Boundary pause requires an all-visible stationary Phase 1 fixed forecast")
        prefetch_budget = self.budget if prefetch_budget is None else prefetch_budget
        decision_overrides = decision_overrides or {}
        if policy not in {"recompute", "longest", "threshold", "local-cost", "fixed", "capacity",
                          "arrived-lookahead", "remaining-wait", "remaining-wait-live",
                          "remaining-wait-fast", "snapshot-score", "snapshot-screened", "adaptive"} | PAIR_POLICIES:
            raise ValueError(f"Unknown policy: {policy}")
        if policy == "fixed" and (boundaries is None or len(boundaries) != n):
            raise ValueError("Fixed policy requires one boundary per request")
        selected = [None] * n
        reserved_at = [None] * n
        handed_off_bytes = [0] * n
        handoff_source = [None] * n
        admitted = {}
        held_reservation = set()
        release_time = None
        host_cache = {}
        host_stats = Counter()
        ready = set(self.initial)
        now = Fraction(0)
        head = arrived = peak = physical_peak = 0
        running_until = transfer = None
        tier_transfers = {}
        tier_factors, tier_records = {}, {}
        compute_factor = transfer_factor = 1
        transfer_record = None
        overlap_time = Counter()
        compute_started = copy_until = transfer_started = None
        active_deps = ()
        completions, starts, head_times, timeline = [], [], [], []
        bytes_by_path = Counter()
        head_since = None
        forecasting = snapshot is not None
        lookahead_replays = 0
        decision_replays = 0
        decision_events = events_processed = pair_decisions = pair_opportunities = 0
        probes = scenario_checks = robust_accepts = pruned_rollouts = 0
        pending = None
        decision_intervals = []
        decision_profile = []
        adaptive_stats = {}
        if adaptive is not None:
            def record_route(label, gap=0, gain=0, changed=False, replays=0, missed=None):
                d = adaptive_stats.setdefault(label, Counter())
                d['decisions'] += 1
                d['certified'] += gap == 0
                d['screened'] += gap is None
                d['changed'] += changed
                d['replays'] += replays
                if gap is not None:
                    d['gap_sum_ms'] += gap
                    d['gap_max_ms'] = max(d['gap_max_ms'], gap)
                d['predicted_gain_sum_ms'] += gain
                if missed is not None:
                    d['audited_screened'] += 1
                    d['screen_misses'] += missed > 0
                    d['screen_missed_gain_sum_ms'] += missed
        if snapshot is not None:
            (now, head, arrived, admitted, ready, running_until, transfer, active_deps) = snapshot[:8]
            held_reservation, release_time, host_cache = snapshot[8:11]
            if self.dynamic:
                copy_until, compute_factor, transfer_factor = snapshot[11:14]
            if self.per_tier:
                tier_transfers, tier_factors = snapshot[14 if self.dynamic else 11:]
            # The snapshot is already past completion and arrival processing at now.
            head_since = now
        lane_slowdown = (future_cost[2] if future_cost is not None and len(future_cost) > 2
                         else self.lane_slowdown)
        estimate = estimate or {}
        execution_factors = estimate if snapshot is not None else {}
        def observed_epoch(path, t):
            return bisect_right(self.io_epochs[path][0], max(0, t - lag)) - 1

        def observed_speed(path, t):
            return self.io_epochs[path][1][observed_epoch(path, t)]

        def path_for(obj):
            if obj in host_cache:
                return "ram_to_device"
            return {"RAM": "ram_to_device", "SSD": "ssd_to_device"}[self.objects[obj]["tier"]]

        def duration(obj, factors, at=None):
            t = now if at is None else at
            path = path_for(obj)
            speed = forecast_speed[path] if forecasting else observed_speed(path, t)
            base = self._retained_times[obj] if obj in host_cache else self._transfer_times[obj]
            return base * factors.get(path, 1) / speed

        def occupied():
            return set().union(*admitted.values()) - self.initial

        def retain(obj):
            if obj not in self._retained_times or obj in host_cache:
                return
            while self.size(host_cache) + self.objects[obj]["bytes"] > self.host_cache_bytes:
                del host_cache[next(iter(host_cache))]
                host_stats["evictions"] += 1
            host_cache[obj] = None
            host_stats["insertions"] += 1

        def resource_signals(i, choices, predicted_run, predicted_transfer):
            # A cheap, static probe at this decision event. The pair rollout
            # still handles future handoffs, compute, and serial I/O exactly.
            reserved = occupied()
            resident = self.initial | ready
            if predicted_run is not None:
                resident |= set(active_deps[:-1])
            in_flight = set()
            wait = 0
            if predicted_transfer is not None:
                end, obj = predicted_transfer
                in_flight.add(obj)
                wait = end - now
            slack = (predicted_run - now) if predicted_run is not None else 0
            times = {}

            def io_time(deps):
                for obj in deps - times.keys():
                    times[obj] = duration(obj, estimate)
                return sum(times[obj] for obj in deps)

            capacity_signal = transfer_signal = False
            for next_c in self.eligible[i + 1]:
                if not next_c:
                    continue
                deps = set(self.candidates[i + 1][next_c])
                missing = deps - resident - in_flight
                fits = set()
                spill = set()
                for c in choices:
                    own = set(self.candidates[i][c])
                    after = reserved | (own - self.initial)
                    if i != head and self.size(after) > self.budget:
                        after = reserved
                    fits.add(self.size(after | (deps - self.initial)) <= self.budget)
                    if missing:
                        backlog = io_time((reserved | own) - resident - in_flight - deps)
                        spill.add(max(0, wait + backlog + io_time(missing) - slack))
                capacity_signal |= len(fits) > 1
                transfer_signal |= bool(spill) and max(spill) - min(spill) > transfer_trigger
            return capacity_signal, transfer_signal

        def choose(i):
            nonlocal lookahead_replays, decision_replays, decision_events, pair_decisions, pair_opportunities, probes
            nonlocal scenario_checks, robust_accepts, pruned_rollouts
            choices = self.eligible[i]
            if i in decision_overrides:
                return decision_overrides[i]
            if policy == "fixed":
                c = boundaries[i]
                if c not in choices:
                    raise ValueError(f"Invalid fixed boundary for request {i}: {c}")
                return c
            if policy == "recompute":
                return 0
            if policy == "longest":
                return max(choices)
            if policy == "threshold":
                return max((c for c in choices if c >= threshold_tokens), default=0)
            if policy in {"remaining-wait-live", "remaining-wait-fast", "snapshot-score",
                          "snapshot-screened", "adaptive"} or policy in PAIR_POLICIES:
                if len(choices) == 1:
                    if adaptive is not None:
                        record_route('single-choice')
                    return choices[0]
                isolated_adaptive = adaptive is not None and arrived == i + 1 and not held_reservation
                if ((isolated_adaptive or policy == "remaining-wait-fast" and not guard) and i == head
                        and running_until is None and transfer is None and not admitted
                        and ready == self.initial):
                    # The one-request rollout has no overlap or shared
                    # reservations; its completion is a serial sum.
                    def isolated(c):
                        deps = self.candidates[i][c]
                        copy = (self.path_time("device_copy", self.objects[deps[-1]]["bytes"])
                                * estimate.get("device_copy", 1)
                                if deps and c < self.lengths[i] and deps[-1] in self.initial else 0)
                        return (self._suffix_times[i][c] * estimate.get("compute", 1) + copy
                                + sum(duration(obj, estimate) for obj in deps if obj not in ready), -c)
                    if isolated_adaptive:
                        record_route('isolated')
                    return min(choices, key=isolated)
                # Estimate remaining work from start times and observed phase,
                # never from the actual completion timestamp of an active phase.
                def remaining(start, predicted):
                    return max(Fraction(0), predicted - (now - start))

                predicted_run = predicted_transfer = None
                if self.dynamic:
                    # With known base work and rates, current progress is
                    # available in the model; future rates are recomputed.
                    predicted_run, predicted_transfer = running_until, transfer
                elif running_until is not None:
                    suffix = self._suffix_times[head][selected[head]] * estimate.get("compute", 1)
                    if now < copy_until:
                        copy = self.path_time("device_copy", self.objects[active_deps[-1]]["bytes"])
                        predicted_run = now + remaining(compute_started, copy * estimate.get("device_copy", 1)) + suffix
                    else:
                        predicted_run = now + remaining(copy_until, suffix)
                if transfer is not None and not self.dynamic:
                    obj = transfer[1]
                    predicted_transfer = (now + remaining(transfer_started, duration(obj, estimate, transfer_started)), obj)
                pair = policy in PAIR_POLICIES and i + 1 < arrived
                exposure_policies = {"two-request-exposed-io", "two-request-exposed-floor", "two-request-stable-io",
                                     "two-request-robust"}
                if pair and policy in exposure_policies:
                    probes += 1
                if pair and policy == "two-request-stable-io":
                    pair = all((epoch := observed_epoch(path, now)) == 0
                               or now - (self.io_epochs[path][0][epoch] + lag) >= stable for path in IO_PATHS)
                if pair and policy in exposure_policies:
                    resident = ready | (set(active_deps[:-1]) if predicted_run is not None else set())
                    queued = occupied() - resident
                    in_flight = 0
                    if predicted_transfer is not None:
                        end, obj = predicted_transfer
                        in_flight = end - now
                        resident.add(obj)
                        queued.discard(obj)
                    slack = predicted_run - now if predicted_run is not None else 0
                    common = in_flight + sum(duration(obj, estimate) for obj in queued)
                    io = {obj: duration(obj, estimate) for obj in set().union(*(
                        self.candidates[i][c] for c in choices)) - resident - queued}
                    incremental = [sum(io.get(obj, 0) for obj in self.candidates[i][c]) for c in choices]
                    exposed = [max(0, common + cost - slack) for cost in incremental]
                    pair = max(exposed) - min(exposed) > transfer_trigger
                    if policy != "two-request-exposed-io":
                        pair = pair and max(incremental) - min(incremental) > 2 * transfer_trigger
                if pair and policy == "two-request-checkpoint-floor":
                    probes += 1
                    checkpoints = {self.candidates[i][c][-1] for c in choices if c} - ready
                    pair = any(duration(obj, estimate) >= 3 * transfer_trigger for obj in checkpoints)
                if pair and policy in {"two-request-backlog", "two-request-backlog-io"}:
                    probes += 1
                    # A running head alone is not evidence of sustained pressure.
                    # Require at least two waiting requests and an aged waiter.
                    pair = arrived - head >= 3 and now - self.arrivals[head + 1] >= transfer_trigger
                if pair and policy in {"two-request-io-floor", "two-request-backlog-io"}:
                    probes += 1
                    # Exclude inexpensive transfers before evaluating the next request.
                    pending_objects = set().union(*(set(self.candidates[i][c]) for c in choices)) - ready
                    pair = sum(duration(obj, estimate) for obj in pending_objects) > 2 * transfer_trigger
                pair_opportunities += pair
                if pair and policy != "two-request":
                    capacity, transfer_pressure = resource_signals(i, choices, predicted_run, predicted_transfer)
                    pair = {"two-request-capacity": capacity, "two-request-transfer": transfer_pressure,
                            "two-request-either": capacity or transfer_pressure,
                            "two-request-shortlist": transfer_pressure,
                            "two-request-io-floor": transfer_pressure,
                            "two-request-backlog": transfer_pressure,
                            "two-request-backlog-io": transfer_pressure,
                            "two-request-checkpoint-floor": transfer_pressure,
                            "two-request-exposed-io": transfer_pressure,
                            "two-request-exposed-floor": transfer_pressure,
                            "two-request-stable-io": transfer_pressure,
                            "two-request-robust": transfer_pressure}[policy]
                pair_decisions += pair
                current_speeds = {path: observed_speed(path, now) for path in IO_PATHS}

                details = {}
                observable = {}
                if observe_candidates is not None or score_candidate is not None or adaptive is not None:
                    reserved = occupied()
                    active_kv = set(active_deps[:-1]) if predicted_run is not None else set()
                    flights = tier_transfers.values() if self.per_tier else (
                        (predicted_transfer,) if predicted_transfer is not None else ())
                    flight_end = max((t for t, _ in flights), default=now)
                    flying = {obj for _, obj in flights}
                    queued = reserved - ready - active_kv - flying
                    slack = predicted_run - now if predicted_run is not None else 0
                    debt = sum(duration(obj, estimate) for obj in queued)
                    charged = reserved | held_reservation
                    free = self.budget - self.size(charged)
                    queued_compute = sum(
                        self._suffix_times[j][selected[j]] * estimate.get("compute", 1)
                        for j in range(head + int(predicted_run is not None), i)
                        if selected[j] is not None)
                    for c in choices:
                        deps = set(self.candidates[i][c])
                        extra = self.size(deps - charged - self.initial)
                        fits = i > head and extra <= free
                        available = ready | (active_kv if fits else set())
                        missing = deps - available
                        incremental = missing - queued - flying
                        if not fits:
                            # No new reservation: do not rely on another
                            # request keeping its objects until our turn.
                            incremental = missing - flying
                        by_path = {path: sum(duration(obj, estimate) for obj in incremental
                                             if path_for(obj) == path) for path in IO_PATHS}
                        serial_by_path = {path: sum(duration(obj, estimate) for obj in deps - self.initial
                                                    if path_for(obj) == path) for path in IO_PATHS}
                        observable[c] = dict(
                            serial_compute_ms=self._suffix_times[i][c] * estimate.get("compute", 1),
                            serial_copy_ms=(self.path_time("device_copy", self.objects[self.candidates[i][c][-1]]["bytes"])
                                            * estimate.get("device_copy", 1)
                                            if c and c < self.lengths[i] and self.candidates[i][c][-1] in self.initial else 0),
                            serial_io_ms=(max(serial_by_path.values()) if self.per_tier
                                          else sum(serial_by_path.values())),
                            head_distance=i - head, arrival_age_ms=now - self.arrivals[i],
                            head_slack_ms=slack, prefetch_fits=fits,
                            committed_ahead_compute_ms=queued_compute,
                            extra_reserve_bytes=extra, free_reserve_bytes=free,
                            ready_bytes=self.size(deps & ready),
                            active_kv_bytes=self.size(deps & active_kv),
                            active_new_kv_bytes=self.size(deps & (active_kv - ready)),
                            active_source_index=head if active_kv else None,
                            shared_reservation_bytes=self.size(deps & reserved),
                            missing_bytes=self.size(missing),
                            incremental_ram_ms=by_path["ram_to_device"],
                            incremental_ssd_ms=by_path["ssd_to_device"],
                            inflight_ms=flight_end - now, queue_debt_ms=debt,
                            checkpoint_tier=("RAM" if self.candidates[i][c][-1] in host_cache else
                                             self.objects[self.candidates[i][c][-1]]["tier"]) if c else "",
                        )
                fallback = None
                horizon = queue_horizon
                estimated = None
                estimated_challenger = None
                if score_candidate is not None:
                    if screen_mode is None:
                        return min(choices, key=lambda c: (score_candidate(observable[c]), -c))
                    def serial(c):
                        d = observable[c]
                        return d["serial_compute_ms"] + d["serial_copy_ms"] + d["serial_io_ms"], -c
                    fallback = min(choices, key=serial)
                    if screen_selector is not None:
                        view = screen_selector(observable, fallback, arrived - i - 1)
                        if view is None:
                            return fallback
                        if view != "current" and (type(view) is not int or view < 0):
                            raise ValueError("Expected None, current, or a nonnegative queue horizon")
                        horizon = None if view == "current" else view
                        if screen_search in {"estimate", "estimate-cross-tier",
                                             "estimate-cross-tier-guard", "estimate-prefetch-fallback"}:
                            scores = {c: score_candidate(observable[c]) for c in choices}
                            estimated_challenger = min(choices, key=lambda c: (scores[c], -c))
                            if scores[estimated_challenger] >= scores[fallback]:
                                return fallback
                            cross_tier = False
                            if screen_search in {"estimate-cross-tier", "estimate-cross-tier-guard",
                                                 "estimate-prefetch-fallback"}:
                                row = observable[estimated_challenger]
                                cross_tier = bool(
                                    fallback and estimated_challenger
                                    and row["checkpoint_tier"] != observable[fallback]["checkpoint_tier"]
                                    and row["prefetch_fits"]
                                    and (row["incremental_ram_ms"] or row["incremental_ssd_ms"])
                                )
                            if screen_search in {"estimate-cross-tier", "estimate-prefetch-fallback"} and cross_tier:
                                return fallback
                            if screen_search == "estimate-prefetch-fallback" and (
                                    0 < estimated_challenger < fallback
                                    and observable[estimated_challenger]["checkpoint_tier"] ==
                                        observable[fallback]["checkpoint_tier"]
                                    and observable[estimated_challenger]["prefetch_fits"]
                                    and not observable[fallback]["prefetch_fits"]
                                    and observable[estimated_challenger]["incremental_ssd_ms"]):
                                return fallback
                            if not cross_tier:
                                return estimated_challenger if (
                                    scores[fallback] - scores[estimated_challenger] > min_gain
                                ) else fallback
                    else:
                        if screen_search in {"compute", "cutoff", "entry"}:
                            estimated = {c: score_candidate(observable[c]) for c in choices}
                            approximate = min(choices, key=lambda c: (estimated[c], -c))
                        else:
                            approximate = min(choices, key=lambda c: (score_candidate(observable[c]), -c))
                        unsettled = any(c and (d["shared_reservation_bytes"] or d["ready_bytes"] or
                                           (d["head_distance"] > 0 and
                                            (d["queue_debt_ms"] or d["inflight_ms"] or
                                             (not d["prefetch_fits"] and
                                              d["extra_reserve_bytes"] <= prefetch_budget))))
                                    for c, d in observable.items())
                        if approximate == fallback and not (screen_mode == "unsettled" and unsettled):
                            return fallback

                def forecast(plan, n_visible, speeds=current_speeds, *, budget=None, handoff=True, cost=None,
                             cutoff=None):
                    nonlocal decision_replays, decision_events, pruned_rollouts
                    state = (now, head, n_visible, dict(admitted), set(ready), predicted_run,
                             predicted_transfer, active_deps, set(held_reservation), release_time, dict(host_cache))
                    if self.dynamic:
                        state += (copy_until, compute_factor, transfer_factor)
                    if self.per_tier:
                        state += (dict(tier_transfers), dict(tier_factors))
                    run = self._run("fixed", 0, (*selected[:i], *plan), max_combinations, n_visible,
                                    snapshot=state, estimate=estimate, forecast_speed=speeds,
                                    prefetch_budget=budget, active_kv_handoff=handoff, future_cost=cost,
                                    completion_cutoff=cutoff,
                                    lean_rollout=screen_search != "full" or adaptive is not None)
                    decision_replays += 1
                    decision_events += run["events_processed"]
                    if run.get("pruned"):
                        pruned_rollouts += 1
                        return None
                    if observe_candidates is not None:
                        details[plan[0]] = (run["head_ms"], run["compute_start_ms"],
                                            run["prefetch_reserved_ms"], run["active_kv_handoff_bytes"],
                                            run["active_kv_handoff_source"])
                    return run["completion_ms"]

                if adaptive is not None:
                    visible = min(arrived, i + 1 + adaptive.later)
                    for j in range(i, visible):
                        if j not in serial_choices:
                            serial_choices[j] = serial_choice(j)
                    base = serial_choices[i]
                    later = tuple(serial_choices[j] for j in range(i + 1, visible))
                    plan = (*selected[head:i], base, *later)
                    factor = estimate.get('compute', 1)
                    work = [self._suffix_times[j][c] * factor for j, c in enumerate(plan, head)]
                    if predicted_run is not None:
                        work[0] = (predicted_run - now) / compute_factor
                    offsets = []
                    time = now
                    for left in work:
                        time += left
                        offsets.append(time)
                    lower = {c: tuple(end + (self._suffix_times[i][c] - self._suffix_times[i][base]) * factor
                                      if j >= i else end for j, end in enumerate(offsets, head))
                             for c in choices}
                    rank = {}
                    for c, row in observable.items():
                        io = row['incremental_ram_ms'] + row['incremental_ssd_ms']
                        slack = row['head_slack_ms'] + row['committed_ahead_compute_ms']
                        wait = max(0, io + row['inflight_ms'] + row['queue_debt_ms'] - slack)
                        if not row['prefetch_fits']:
                            wait = io + max(0, row['inflight_ms'] - slack)
                        rank[c] = row['serial_compute_ms'] + row['serial_copy_ms'] + (
                            wait if row['missing_bytes'] else 0)
                    before = decision_replays
                    boundary, label, gap, gain, missed = adaptive_select(
                        observable, base, self.budget, adaptive, lower,
                        lambda c: forecast((c, *later), visible), rank)
                    record_route(label, gap, gain, boundary != base, decision_replays - before, missed)
                    return boundary

                if pair:
                    next_choices = self.eligible[i + 1]
                    if policy == "two-request-shortlist" and len(next_choices) > 2:
                        # Retain recompute and the best isolated nonzero option.
                        # This avoids extra rollouts just to construct a shortlist.
                        def next_cost(c):
                            deps = self.candidates[i + 1][c]
                            copy = (self.path_time("device_copy", self.objects[deps[-1]]["bytes"])
                                    * estimate.get("device_copy", 1)
                                    if deps[-1] in self.initial and c < self.lengths[i + 1] else 0)
                            return (self.recompute(i + 1, c) * estimate.get("compute", 1) + copy
                                    + sum(duration(obj, estimate) for obj in deps if obj not in self.initial), -c)

                        next_choices = (0, min((c for c in next_choices if c), key=next_cost))
                    scores = {c: min(sum(forecast((c, next_c), i + 2)[-2:])
                                     for next_c in next_choices) for c in choices}
                else:
                    visible = (i + 1 if horizon is None else
                               min(arrived, i + 1 + horizon) if horizon else arrived)
                    if self.host_cache_bytes:
                        later = tuple(serial_choice(j) for j in range(i + 1, visible))
                    else:
                        for j in range(i + 1, visible):
                            if j not in serial_choices:
                                serial_choices[j] = serial_choice(j)
                        later = tuple(serial_choices[j] for j in range(i + 1, visible))
                    def objective(c, cost=None, cutoff=None):
                        ends = forecast((c, *later), visible, cost=cost, cutoff=cutoff)
                        if ends is None:
                            return None
                        if horizon is None:
                            return ends[-1]
                        # All unfinished requests in view, including committed work ahead.
                        # Subtracting their arrival timestamps is constant across trials.
                        return sum(ends)
                    if estimated_challenger is not None:
                        # Only the estimate's cross-tier challenger reaches this path.
                        baseline_ends = forecast((fallback, *later), visible)
                        challenger_ends = forecast((estimated_challenger, *later), visible)
                        if any(new > old for old, new in zip(baseline_ends, challenger_ends, strict=True)):
                            return fallback
                        scores = {fallback: sum(baseline_ends), estimated_challenger: sum(challenger_ends)}
                    elif screen_search not in {"compute", "cutoff", "entry"} or horizon is None:
                        scores = {c: objective(c) for c in choices}
                    else:
                        # Keep an exact fallback score for the gain threshold.
                        order = sorted((c for c in choices if c != fallback),
                                       key=lambda c: (estimated[c] if estimated is not None else
                                                      score_candidate(observable[c]), -c))
                        floor = [0] * (visible + 1)
                        tail = (*selected[head:i], fallback, *later)
                        factor = estimate.get("compute", 1)
                        for j in range(visible - 1, head - 1, -1):
                            floor[j] = floor[j + 1] + (
                                (visible - j) * self._suffix_times[j][tail[j - head]] * factor)
                        initial_floor = 0
                        if screen_search == "entry":
                            jobs = visible - head
                            initial_floor = jobs * now + floor[head]
                            if predicted_run is not None:
                                work = self._suffix_times[head][selected[head]] * factor
                                initial_floor += jobs * ((predicted_run - now) / compute_factor - work)
                        scores = {fallback: objective(fallback)}
                        best = fallback
                        for c in order:
                            delta = (visible - i) * factor * (
                                self._suffix_times[i][c] - self._suffix_times[i][fallback])
                            if screen_search == "entry":
                                lower = initial_floor + delta
                                if (lower, -c) >= (scores[best], -best):
                                    pruned_rollouts += 1
                                    continue
                                cutoff = None
                            else:
                                cutoff = (scores[best], c < best, floor, i, delta,
                                          screen_search == "cutoff")
                            s = objective(c, cutoff=cutoff)
                            if s is not None:
                                scores[c] = s
                                if (s, -c) < (scores[best], -best):
                                    best = c
                fastest = min(scores, key=lambda c: (scores[c], -c))
                if fallback is not None:
                    if screen_observer is not None:
                        screen_observer(observable, fallback, arrived - i - 1, scores[fallback] - scores[fastest])
                    if scores[fallback] - scores[fastest] <= min_gain:
                        return fallback
                    if future_cost_cases:
                        for cost in future_cost_cases.values():
                            tested = objective(fastest, cost)
                            baseline = objective(fallback, cost)
                            if baseline <= tested:
                                return fallback
                if observe_candidates is not None:
                    costs = {}
                    for c in choices:
                        queue_head, start, reservation, handoff_bytes, source = details[c]
                        costs[c] = dict(
                            compute_ms=scores[c] - start,
                            head_offset_ms=queue_head - now,
                            exposed_io_and_copy_ms=start - queue_head,
                            prefetch_reservation_offset_ms=reservation - now if reservation is not None else None,
                            active_kv_handoff_bytes=handoff_bytes,
                            active_kv_handoff_source=source,
                            finish_offset_ms=scores[c] - now,
                            **observable[c],
                        )
                        if intervention_budget is not None:
                            costs[c]["intervention_finish_ms"] = {
                                "baseline": scores[c] - now,
                                "larger_budget": forecast((c,), i + 1, budget=intervention_budget)[-1] - now,
                                "without_active_handoff": forecast((c,), i + 1, handoff=False)[-1] - now,
                                "both": forecast((c,), i + 1, budget=intervention_budget, handoff=False)[-1] - now,
                            }
                        if future_cost_cases:
                            costs[c]["future_cost_finish_ms"] = {
                                "nominal": scores[c] - now,
                                **{name: forecast((c,), i + 1, cost=cost)[-1] - now
                                   for name, cost in future_cost_cases.items()},
                            }
                    observe_candidates(i, now, arrived - head, costs)
                if pair and policy == "two-request-robust":
                    fallback = min(choices, key=lambda c: (forecast((c,), i + 1)[-1], -c))
                    initial_cost = pair_cost * Fraction(4, 3) + screen + probe_cost
                    if fastest == fallback or scores[fallback] - scores[fastest] <= initial_cost:
                        return fallback
                    ranges = []
                    for path in IO_PATHS:
                        end = observed_epoch(path, now)
                        start = observed_epoch(path, now - 60)
                        history = self.io_epochs[path][1][start:end + 1]
                        ranges.append(tuple(dict.fromkeys((min(history), max(history), current_speeds[path]))))
                    tested = 0
                    margin = scores[fallback] - scores[fastest]
                    for rates in product(*ranges):
                        speeds = dict(zip(IO_PATHS, rates))
                        if speeds == current_speeds:
                            continue
                        scenario_checks += 1
                        tested += 1
                        alt, base = (min(sum(forecast((c, next_c), i + 2, speeds)[-2:])
                                         for next_c in self.eligible[i + 1]) for c in (fastest, fallback))
                        margin = min(margin, base - alt)
                        if margin <= initial_cost + pair_cost * tested:
                            return fallback
                    robust_accepts += 1
                    return fastest
                if not guard:
                    return fastest
                # When candidates are effectively tied, favor smaller incremental
                # reservations; identical ready objects incur no extra transfer.
                near = [c for c in choices if scores[c] <= scores[fastest] + guard]
                resident = occupied() | self.initial
                return min(near, key=lambda c: (self.size(set(self.candidates[i][c]) - resident),
                                                scores[c], -c))
            if policy == "remaining-wait":
                if len(choices) == 1:
                    return choices[0]
                # Prior choices are committed in FIFO order. Replay only through
                # this request to price its residual wait without joint search.
                committed = tuple(selected[:i])
                decision_replays += len(choices)

                def completion(c):
                    run = self._run("fixed", 0, (*committed, c), max_combinations, i + 1)
                    return run["completion_ms"][-1], -c

                return min(choices, key=completion)
            if policy == "arrived-lookahead":
                if len(choices) == 1:
                    return choices[0]
                # Deterministic replay reconstructs the current state from the
                # arrived prefix and committed choices. Unseen arrivals cannot
                # alter the past; ties between arrivals are processed together.
                domains = [(selected[j],) if selected[j] is not None else self.eligible[j]
                           for j in range(arrived)]
                count = prod(map(len, domains))
                if count > max_combinations:
                    raise ValueError(f"Arrived-prefix search requires {count} combinations "
                                     f"(limit {max_combinations})")
                lookahead_replays += count
                best = min(product(*domains), key=lambda plan: (
                    self._run("fixed", 0, plan, max_combinations, arrived)["response_sum_ms"],
                    tuple(-c for c in plan),
                ))
                return best[i]

            return serial_choice(i)

        def serial_choice(i):
            def score(c):
                deps = set(self.candidates[i][c])
                missing = deps - self.initial
                checkpoint = self.candidates[i][c][-1] if c else None
                copy = 0
                if 0 < c < self.lengths[i] and checkpoint in self.initial:
                    copy = self.path_time("device_copy", self.objects[checkpoint]["bytes"])
                total = self.recompute(i, c) + copy + sum(map(self.transfer_time, missing))
                if self.host_cache_bytes:
                    total = self.recompute(i, c) + copy + sum(duration(obj, {}) for obj in missing)
                if self.per_tier:
                    total = self.recompute(i, c) + copy + max(
                        sum(duration(obj, estimate) for obj in missing if path_for(obj) == path)
                        for path in IO_PATHS)
                if policy in {"local-cost", "adaptive"} and estimate:
                    total = (self.recompute(i, c) * estimate.get("compute", 1)
                             + copy * estimate.get("device_copy", 1)
                             + sum(duration(obj, estimate) for obj in missing))
                if policy == "capacity" and running_until is not None:
                    union = occupied() | missing
                    if self.size(union) <= self.budget:
                        total = self.recompute(i, c) + copy
                return total, -c

            return min(self.eligible[i], key=score)

        serial_choices = {}

        def begin_decision(i):
            nonlocal pending
            if offline_probe is not None and boundaries[i] is None:
                if len(self.eligible[i]) == 1:
                    selected[i] = self.eligible[i][0]
                    return
                raise OfflineBranch(offline_probe(i, now, head, completions, ready, running_until,
                                                  compute_factor, transfer, transfer_factor, active_deps))
            opportunities, pairs, prior_probes, checks = pair_opportunities, pair_decisions, probes, scenario_checks
            timed = profile_decisions or inject_measured
            start = perf_counter_ns() if timed else 0
            boundary = choose(i)
            elapsed = perf_counter_ns() - start if timed else 0
            if timed:
                kind = ("pair" if pair_decisions > pairs else
                        "screen" if pair_opportunities > opportunities else
                        "probe" if probes > prior_probes else "single")
                decision_profile.append((kind, elapsed))
            delay = (screen * (pair_opportunities - opportunities)
                     + pair_cost * (pair_decisions - pairs + scenario_checks - checks)
                     + probe_cost * (probes - prior_probes))
            if policy == "two-request-robust":
                delay += pair_cost * Fraction(pair_decisions - pairs, 3)
            if inject_measured:
                delay += Fraction(elapsed, 1_000_000)
            if delay:
                pending = (now + delay, i, boundary)
                decision_intervals.append((now, now + delay, i))
            else:
                selected[i] = boundary

        def next_event():
            nonlocal running_until, transfer, compute_factor, transfer_factor
            copy_end = busy_path = None
            lane_prefill = {}
            if self.dynamic:
                new_compute, new_transfer = 1, 1
                if running_until is not None and tier_transfers:
                    if now < copy_until:
                        copy_end = copy_until
                    elif len(tier_transfers) == 2:
                        busy_path = "both_lanes_prefill"
                        new_compute = self.joint_prefill["compute"]
                        lane_prefill = self.joint_prefill
                    else:
                        busy_path = next(iter(tier_transfers))
                        new_compute, factor = self.overlap.get(busy_path, (1, 1))
                        lane_prefill = {busy_path: factor}
                elif running_until is not None and transfer is not None:
                    path = path_for(transfer[1])
                    if path in self.overlap:
                        if now < copy_until:
                            copy_end = copy_until
                        else:
                            busy_path = path
                            new_compute, new_transfer = self.overlap[path]
                if running_until is not None and new_compute != compute_factor:
                    running_until = now + (running_until - now) / compute_factor * new_compute
                if transfer is not None and new_transfer != transfer_factor:
                    transfer = (now + (transfer[0] - now) / transfer_factor * new_transfer, transfer[1])
                    if transfer_record is not None:
                        transfer_record["end_ms"] = transfer[0]
                compute_factor, transfer_factor = new_compute, new_transfer
            if lane_slowdown or tier_factors or lane_prefill:
                # Convert remaining scheduled time back to solo work before
                # applying the new rate. No object restarts at a transition.
                for path, (end, obj) in tier_transfers.items():
                    factor = lane_slowdown.get(path, 1) if len(tier_transfers) == 2 else 1
                    factor *= lane_prefill.get(path, 1)
                    old = tier_factors.get(path, 1)
                    if old != factor:
                        end = now + (end - now) / old * factor
                        tier_transfers[path] = (end, obj)
                        if path in tier_records:
                            tier_records[path]["end_ms"] = end
                    tier_factors[path] = factor
            end = min(t for t in (running_until, transfer[0] if transfer else None, copy_end, release_time,
                                  *(item[0] for item in tier_transfers.values()),
                                  self.arrivals[arrived] if arrived < n else None,
                                  pending[0] if pending else None) if t is not None)
            if busy_path is not None:
                overlap_time[busy_path] += end - now
            if len(tier_transfers) == 2:
                overlap_time["both_lanes"] += end - now
            return end

        def cannot_beat():
            target, tie_loses, floor, replaced, delta, demand_bound = completion_cutoff
            jobs = n - head
            lower = sum(completions) + jobs * now + floor[head] + (delta if head <= replaced else 0)
            if running_until is not None:
                work = self._suffix_times[head][boundaries[head]] * execution_factors.get("compute", 1)
                lower += jobs * ((running_until - now) / compute_factor - work)
            elif demand_bound and not self.per_tier and not self.host_cache_bytes:
                missing = set(self.candidates[head][boundaries[head]]) - ready
                if missing:
                    if transfer is not None:
                        lower += jobs * ((transfer[0] - now) / transfer_factor)
                        missing.discard(transfer[1])
                    lower += jobs * sum(duration(obj, execution_factors) for obj in missing)
            return lower > target or (tie_loses and lower == target)

        while head < n:
            events_processed += 1
            resume = snapshot is not None
            snapshot = None
            if release_time == now:
                held_reservation = set()
                release_time = None
            if not resume and transfer is not None and transfer[0] == now:
                ready.add(transfer[1])
                if self.host_cache_bytes:
                    retain(transfer[1])
                transfer = None
                transfer_factor, transfer_record = 1, None
            if not resume and tier_transfers:
                for path in tuple(tier_transfers):
                    end, obj = tier_transfers[path]
                    if end == now:
                        ready.add(obj)
                        del tier_transfers[path]
                        tier_factors.pop(path, None)
                        tier_records.pop(path, None)
            advanced = not resume and running_until == now
            if advanced:
                completions.append(now)
                held_reservation = set()
                release_time = None
                head += 1
                running_until = None
                compute_factor = 1
                head_since = None
                if head == n:
                    break
            if pause_at_head is not None and advanced and head == pause_at_head:
                # Preserve the same post-completion event phase expected by
                # snapshot resumption; containers must not alias the caller.
                state = (now, head, arrived, dict(admitted), set(ready), running_until,
                         transfer, active_deps, set(held_reservation), release_time, dict(host_cache))
                return {"completion_ms": completions, "events_processed": events_processed,
                        "snapshot": state}
            if completion_cutoff is not None and (resume or advanced) and cannot_beat():
                return {"pruned": True, "events_processed": events_processed}
            while arrived < n and self.arrivals[arrived] <= now:
                arrived += 1
            if pending is not None and pending[0] == now:
                _, i, selected_boundary = pending
                selected[i] = selected_boundary
                pending = None
            if head < arrived and head_since is None:
                head_since = now
            if pending is not None:
                now = next_event()
                continue
            deps = ()
            if head < arrived:
                if head_since is None:
                    head_since = now
                if selected[head] is None:
                    begin_decision(head)
                    if pending is not None:
                        now = next_event()
                        continue
                deps = self.candidates[head][selected[head]]
                if running_until is None and ready.issuperset(deps):
                    reservation = admitted.pop(head, None)
                    if self.release_at != "start" and reservation:
                        held_reservation = reservation
                        if self.release_at == "timer":
                            release_time = now + self.release_delay
                    shared = bool(deps) and (deps[-1] in self.initial or deps[-1] in occupied())
                    suffix_times = future_cost[0] if future_cost is not None else self._suffix_times
                    suffix = suffix_times[head][selected[head]] * execution_factors.get("compute", 1)
                    continuation = selected[head] < self.lengths[head]
                    copy = (self.path_time("device_copy", self.objects[deps[-1]]["bytes"])
                            * execution_factors.get("device_copy", 1)
                            if shared and continuation else 0)
                    if shared and continuation and not lean_rollout:
                        size = self.objects[deps[-1]]["bytes"]
                        bytes_by_path["device_copy"] += size
                        timeline.append({"path": "device_copy", "object": deps[-1],
                                         "request": self.requests[head]["id"], "mode": "handoff",
                                         "start_ms": now, "end_ms": now + copy, "bytes": size})
                    active_deps = deps
                    compute_started, copy_until = now, now + copy
                    if not lean_rollout:
                        starts.append(now + copy)
                        head_times.append(head_since)
                    running_until = now + copy + suffix
                    if not lean_rollout:
                        physical_peak = max(physical_peak, self.fixed + self.size(occupied()) +
                                            (self.run_bytes[head] - self.size(
                                                set(deps[:-1]) & (occupied() | self.initial))
                                             if continuation else
                                             self.size(set(deps) - self.initial - occupied())))
                    ready.intersection_update(occupied() | self.initial)
                    if running_until == now:
                        continue
                for i in range(head + 1, min(arrived, head + 1 + self.window)):
                    if selected[i] is None:
                        begin_decision(i)
                        if pending is not None:
                            break
                    future = set(self.candidates[i][selected[i]]) - self.initial
                    if i not in admitted and future and self.size(occupied() | held_reservation | future) <= prefetch_budget:
                        admitted[i] = future
                        if running_until is not None and active_kv_handoff:
                            reused = future.intersection(active_deps[:-1]) - ready
                            if forecasting and not lean_rollout:
                                handed_off_bytes[i] = sum(self.objects[obj]["bytes"] for obj in reused)
                                if reused:
                                    handoff_source[i] = head
                            ready.update(reused)
                        if not lean_rollout:
                            reserved_at[i] = now

            if pending is not None:
                now = next_event()
                continue

            if not lean_rollout:
                occupied_now = occupied()
                peak = max(peak, self.size(occupied_now | held_reservation))
                active_bytes = (
                    self.run_bytes[head] - self.size(set(active_deps[:-1]) & (occupied_now | self.initial))
                    if running_until is not None else self.size(set(deps) - self.initial - occupied_now)
                )
                physical_peak = max(physical_peak, self.fixed + self.size(occupied_now) + active_bytes)
            if self.per_tier or transfer is None:
                order = []
                if running_until is None:
                    order.extend((head, obj, "demand") for obj in deps)
                for i in sorted(admitted):
                    order.extend((i, obj, "prefetch") for obj in self.candidates[i][selected[i]])
                for i, obj, mode in order:
                    if obj not in ready:
                        path = path_for(obj)
                        if path in tier_transfers:
                            continue
                        service = (duration(obj, execution_factors) if forecasting or self.host_cache_bytes else
                                   self._actual_transfer_times[obj] / self.io_epochs[path_for(obj)][1][
                                       bisect_right(self.io_epochs[path_for(obj)][0], now) - 1])
                        end = now + service
                        if future_cost is not None:
                            service *= future_cost[1].get(path, 1)
                            end = now + service
                        if obj in host_cache:
                            host_cache[obj] = host_cache.pop(obj)
                            host_stats["hits"] += 1
                        if not lean_rollout:
                            size = self.objects[obj]["bytes"]
                            bytes_by_path[path] += size
                            timeline.append({"path": path, "object": obj, "request": self.requests[i]["id"],
                                             "mode": mode, "start_ms": now, "end_ms": end, "bytes": size})
                        if self.per_tier:
                            tier_transfers[path] = (end, obj)
                            if not lean_rollout:
                                tier_records[path] = timeline[-1]
                            continue
                        transfer = (end, obj)
                        transfer_record = None if lean_rollout else timeline[-1]
                        transfer_started = now
                        break
            now = next_event()

        if forecasting:
            if lean_rollout:
                return {"completion_ms": completions, "events_processed": events_processed}
            return {"completion_ms": completions, "events_processed": events_processed,
                    "head_ms": head_times[-1], "compute_start_ms": starts[-1],
                    "prefetch_reserved_ms": reserved_at[-1],
                    "active_kv_handoff_bytes": handed_off_bytes[-1],
                    "active_kv_handoff_source": handoff_source[-1]}
        response = [end - arrival for end, arrival in zip(completions, self.arrivals)]
        return {
            "request_ids": [r["id"] for r in self.requests[:n]],
            "boundaries_tokens": selected,
            "prefetch_reserved_at_ms": reserved_at,
            "peak_prefetch_bytes": peak,
            "peak_accounted_device_bytes": physical_peak,
            "completion_ms": completions,
            "queue_wait_ms": [h - a for h, a in zip(head_times, self.arrivals)],
            "head_to_compute_ms": [s - h for s, h in zip(starts, head_times)],
            "response_sum_ms": sum(response),
            "response_mean_ms": sum(response) / n,
            "transferred_bytes_by_path": dict(bytes_by_path),
            "recomputed_tokens": sum(length - c for length, c in zip(self.lengths, selected)),
            "transfers": timeline,
            **({"overlap_ms_by_path": dict(overlap_time)} if self.dynamic else {}),
            **({"lane_overlap_ms": overlap_time["both_lanes"]} if self.per_tier else {}),
            **({"staged_checkpoint_cache": dict(host_stats)} if self.host_cache_bytes else {}),
            **({"lookahead_replays": lookahead_replays} if policy == "arrived-lookahead" else {}),
            **({"decision_replays": decision_replays} if policy == "remaining-wait" else {}),
            **({"decision_replays": decision_replays, "decision_events": decision_events}
               if policy in {"remaining-wait-live", "remaining-wait-fast", "snapshot-screened", "adaptive"}
               or policy in PAIR_POLICIES else {}),
            **({"adaptive_routes": adaptive_stats} if adaptive is not None else {}),
            **({"pruned_rollouts": pruned_rollouts} if screen_search in {"compute", "cutoff", "entry"} else {}),
            **({"pair_decisions": pair_decisions, "pair_opportunities": pair_opportunities}
               if policy in PAIR_POLICIES else {}),
            **({"io_probes": probes} if policy in PROBE_POLICIES else {}),
            **({"scenario_checks": scenario_checks, "robust_accepts": robust_accepts}
               if policy == "two-request-robust" else {}),
            "events_processed": events_processed,
            **({"decision_intervals_ms": decision_intervals} if screen or pair_cost or probe_cost or inject_measured else {}),
            **({"decision_profile_ns": decision_profile} if profile_decisions or inject_measured else {}),
        }

    def offline(self, max_combinations=10000):
        count = prod(map(len, self.eligible))
        if count > max_combinations:
            raise ValueError(f"Offline search requires {count} combinations (limit {max_combinations})")
        best, ties = None, []
        for boundaries in product(*self.eligible):
            run = self.run("fixed", boundaries=boundaries)
            if best is None or run["response_sum_ms"] < best["response_sum_ms"]:
                best, ties = run, []
            if run["response_sum_ms"] == best["response_sum_ms"]:
                ties.append(list(boundaries))
        return {**best, "offline_optima": ties, "combinations_evaluated": count}
