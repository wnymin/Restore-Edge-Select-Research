"""Phase 1 deterministic event simulation. All clock arithmetic uses fractions."""

from collections import Counter
from fractions import Fraction
from itertools import product
from math import prod


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
        self.cost = trace["cost"]
        self.a = Fraction(*self.cost["recompute"]["a_ms_per_token_rational"])
        self.b = Fraction(*self.cost["recompute"]["b_ms_per_pair_rational"])
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
        if self.a < 0 or self.b < 0:
            raise ValueError("Recompute coefficients must be nonnegative")
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

    def size(self, ids):
        return sum(self.objects[obj]["bytes"] + self.alignment for obj in ids)

    def path_time(self, path, size):
        p = self.cost[path]
        startup, bandwidth = number(p["startup_ms"]), number(p["bytes_per_ms"])
        if startup < 0 or bandwidth <= 0:
            raise ValueError(f"Invalid transfer cost: {path}")
        return startup + size / bandwidth

    def transfer_time(self, obj):
        item = self.objects[obj]
        return self.path_time({"RAM": "ram_to_device", "SSD": "ssd_to_device"}[item["tier"]],
                              item["bytes"])

    def recompute(self, i, c):
        length = self.lengths[i]
        return self.a * (length - c) + self.b * ((length - c) * (length + c + 1) // 2)

    def run(self, policy="local-cost", *, threshold_tokens=0, boundaries=None):
        if policy not in {"recompute", "longest", "threshold", "local-cost", "fixed", "capacity"}:
            raise ValueError(f"Unknown policy: {policy}")
        n = len(self.requests)
        if policy == "fixed" and (boundaries is None or len(boundaries) != n):
            raise ValueError("Fixed policy requires one boundary per request")
        selected = [None] * n
        reserved_at = [None] * n
        admitted = {}
        ready = set(self.initial)
        now = Fraction(0)
        head = arrived = peak = physical_peak = 0
        running_until = transfer = None
        active_deps = ()
        completions, starts, head_times, timeline = [], [], [], []
        bytes_by_path = Counter()
        head_since = None

        def occupied():
            return set().union(*admitted.values()) - self.initial

        def choose(i):
            choices = self.eligible[i]
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

            def score(c):
                deps = set(self.candidates[i][c])
                missing = deps - self.initial
                checkpoint = self.candidates[i][c][-1] if c else None
                copy = 0
                if 0 < c < self.lengths[i] and checkpoint in self.initial:
                    copy = self.path_time("device_copy", self.objects[checkpoint]["bytes"])
                total = self.recompute(i, c) + copy + sum(map(self.transfer_time, missing))
                if policy == "capacity" and running_until is not None:
                    union = occupied() | missing
                    if self.size(union) <= self.budget:
                        total = self.recompute(i, c) + copy
                return total, -c

            return min(choices, key=score)

        while head < n:
            if transfer is not None and transfer[0] == now:
                ready.add(transfer[1])
                transfer = None
            if running_until == now:
                completions.append(now)
                head += 1
                running_until = None
                head_since = None
                if head == n:
                    break
            while arrived < n and self.arrivals[arrived] <= now:
                arrived += 1
            deps = ()
            if head < arrived:
                if head_since is None:
                    head_since = now
                if selected[head] is None:
                    selected[head] = choose(head)
                deps = self.candidates[head][selected[head]]
                if running_until is None and ready.issuperset(deps):
                    admitted.pop(head, None)
                    shared = bool(deps) and (deps[-1] in self.initial or deps[-1] in occupied())
                    suffix = self.recompute(head, selected[head])
                    continuation = selected[head] < self.lengths[head]
                    copy = (self.path_time("device_copy", self.objects[deps[-1]]["bytes"])
                            if shared and continuation else 0)
                    if shared and continuation:
                        size = self.objects[deps[-1]]["bytes"]
                        bytes_by_path["device_copy"] += size
                        timeline.append({"path": "device_copy", "object": deps[-1],
                                         "request": self.requests[head]["id"], "mode": "handoff",
                                         "start_ms": now, "end_ms": now + copy, "bytes": size})
                    active_deps = deps
                    starts.append(now + copy)
                    head_times.append(head_since)
                    running_until = now + copy + suffix
                    physical_peak = max(physical_peak, self.fixed + self.size(occupied()) +
                                        (self.run_bytes[head] - self.size(set(deps[:-1]) & (occupied() | self.initial))
                                         if continuation else
                                         self.size(set(deps) - self.initial - occupied())))
                    ready.intersection_update(occupied() | self.initial)
                    if running_until == now:
                        continue
                for i in range(head + 1, min(arrived, head + 1 + self.window)):
                    if selected[i] is None:
                        selected[i] = choose(i)
                    future = set(self.candidates[i][selected[i]]) - self.initial
                    if i not in admitted and future and self.size(occupied() | future) <= self.budget:
                        admitted[i] = future
                        if running_until is not None:
                            ready.update(future.intersection(active_deps[:-1]))
                        reserved_at[i] = now

            occupied_now = occupied()
            peak = max(peak, self.size(occupied_now))
            active_bytes = (
                self.run_bytes[head] - self.size(set(active_deps[:-1]) & (occupied_now | self.initial))
                if running_until is not None else self.size(set(deps) - self.initial - occupied_now)
            )
            physical_peak = max(physical_peak, self.fixed + self.size(occupied_now) + active_bytes)
            if transfer is None:
                order = []
                if running_until is None:
                    order.extend((head, obj, "demand") for obj in deps)
                for i in sorted(admitted):
                    order.extend((i, obj, "prefetch") for obj in self.candidates[i][selected[i]])
                for i, obj, mode in order:
                    if obj not in ready:
                        end = now + self.transfer_time(obj)
                        path = {"RAM": "ram_to_device", "SSD": "ssd_to_device"}[self.objects[obj]["tier"]]
                        size = self.objects[obj]["bytes"]
                        bytes_by_path[path] += size
                        timeline.append({"path": path, "object": obj, "request": self.requests[i]["id"],
                                         "mode": mode, "start_ms": now, "end_ms": end, "bytes": size})
                        transfer = (end, obj)
                        break
            events = [t for t in (running_until, transfer[0] if transfer else None,
                                  self.arrivals[arrived] if arrived < n else None) if t is not None]
            now = min(events)

        response = [end - arrival for end, arrival in zip(completions, self.arrivals)]
        return {
            "request_ids": [r["id"] for r in self.requests],
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
