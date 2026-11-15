"""Two-slot prefill counterfactual using the same ordered start and shared I/O rules."""

from dataclasses import dataclass, field
from fractions import Fraction as F


@dataclass
class State:
    now: F = F(0)
    cursor: int = 0
    arrived: int = 0
    active: dict = field(default_factory=dict)
    admitted: dict = field(default_factory=dict)
    ready: set = field(default_factory=set)
    selected: dict = field(default_factory=dict)
    done: dict = field(default_factory=dict)
    reserved: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    flight: tuple | None = None
    batch_ms: F = F(0)
    started: dict = field(default_factory=dict)

    def clone(self):
        return State(self.now, self.cursor, self.arrived, self.active.copy(), self.admitted.copy(),
                     self.ready.copy(), self.selected.copy(), self.done.copy(), self.reserved.copy(),
                     {i: sources.copy() for i, sources in self.sources.items()}, self.flight, self.batch_ms,
                     self.started.copy())


class BatchReplay:
    def __init__(self, exp, rates):
        if exp.per_tier or exp.host_cache_bytes or exp.release_at != "start" or (
                exp._actual_transfer_times != exp._transfer_times or
                any(t != (0,) or s != (1,) for t, s in exp.io_epochs.values())):
            raise ValueError("Batch contrast requires shared stationary I/O and start release")
        self.exp, self.rates = exp, rates
        self.needs = [{c: frozenset(d) for c, d in item.items()} for item in exp.candidates]
        self.paths = {o: "ram_to_device" if item["tier"] == "RAM" else "ssd_to_device"
                      for o, item in exp.objects.items() if o not in exp.initial}
        self.baseline = [min(choices, key=lambda c: (exp.recompute(i, c) +
                         sum(exp._transfer_times[o] for o in self.needs[i][c] - exp.initial) +
                         self.copy_ms(i, c, exp.initial), -c))
                         for i, choices in enumerate(exp.eligible)]

    def copy_ms(self, i, c, shared):
        deps = self.exp.candidates[i][c]
        if deps and c < self.exp.lengths[i] and deps[-1] in shared:
            return self.exp.path_time("device_copy", self.exp.objects[deps[-1]]["bytes"])
        return F(0)

    def rollout(self, state, i, c, later, slots):
        next_state = state.clone()
        next_state.arrived = later
        next_state.selected[i] = c
        for j in range(i + 1, later):
            next_state.selected[j] = self.baseline[j]
        return self.run(slots=slots, state=next_state, limit=later)

    def run(self, *, slots=1, state=None, limit=None, choose=None, fixed=None):
        if slots not in (1, 2):
            raise ValueError("Expected one or two execution slots")
        exp = self.exp
        total = exp.trace["memory"]["device_mib"] * exp.trace["units"]["bytes_per_mib"]
        if exp.fixed + slots * exp.execution + exp.budget > total:
            raise ValueError("Memory budget cannot fit the requested separate execution slots")
        s = State(ready=set(exp.initial)) if state is None else state
        n = len(exp.requests) if limit is None else limit

        def occupied():
            return exp.initial.union(*s.admitted.values())

        def select(i):
            if i not in s.selected:
                s.selected[i] = (fixed[i] if fixed is not None else self.baseline[i] if choose is None
                                 else choose(s, i))
            c = s.selected[i]
            return c, self.needs[i][c]

        while len(s.done) < n:
            while s.arrived < n and exp.arrivals[s.arrived] <= s.now:
                s.arrived += 1
            if s.flight is not None and s.flight[1] == 0:
                s.ready.add(s.flight[0])
                s.flight = None
            for j, (copy, work, _) in tuple(s.active.items()):
                if copy == work == 0:
                    s.done[j] = s.now
                    del s.active[j]
            demand = None
            while s.cursor < s.arrived and len(s.active) < slots:
                i = s.cursor
                c, deps = select(i)
                if not deps <= s.ready:
                    demand = i
                    break
                s.admitted.pop(i, None)
                backing = occupied()
                old_kv = frozenset(exp.candidates[i][c][:-1])
                copy = self.copy_ms(i, c, backing)
                work = exp.recompute(i, c)
                s.started[i] = s.now + copy
                s.cursor += 1
                s.ready.intersection_update(backing)
                if copy or work:
                    # Each slot owns a separate private execution envelope;
                    # no second owner can overwrite an active checkpoint.
                    assert exp.run_bytes[i] <= exp.execution
                    s.active[i] = (copy, work, old_kv)
                else:
                    s.done[i] = s.now
            if len(s.done) == n:
                break
            first = s.cursor + (demand is not None)
            # At full occupancy the next queued request is the first prefetch.
            for i in range(first, min(s.arrived, first + exp.window)):
                c, deps = select(i)
                future = deps - exp.initial
                if i in s.admitted or not future or exp.size((occupied() | future) - exp.initial) > exp.budget:
                    continue
                s.admitted[i] = future
                s.reserved[i] = s.now
                for j, (_, _, kv) in s.active.items():
                    reused = future.intersection(kv) - s.ready
                    if reused:
                        s.sources.setdefault(i, {})[j] = exp.size(reused)
                        s.ready.update(reused)
            if s.flight is None:
                order = ([demand] if demand is not None else []) + sorted(s.admitted)
                obj = next((obj for i in order for obj in exp.candidates[i][s.selected[i]]
                            if obj not in s.ready), None)
                if obj is not None:
                    s.flight = obj, exp._transfer_times[obj]
            work_ids = [i for i, (copy, work, _) in s.active.items() if copy == 0 and work > 0]
            rates = dict.fromkeys(work_ids, F(1))
            if len(work_ids) == 2:
                pair = [(exp.recompute(i, s.selected[i]), exp.lengths[i] - s.selected[i]) for i in work_ids]
                rates = dict(zip(work_ids, self.rates(pair), strict=True))
            transfer_factor = F(1)
            if work_ids and s.flight is not None:
                comp_factor, transfer_factor = exp.overlap.get(self.paths[s.flight[0]], (1, 1))
                rates = {i: rate / comp_factor for i, rate in rates.items()}
            steps = [exp.arrivals[s.arrived] - s.now] if s.arrived < n else []
            if s.flight is not None:
                steps.append(s.flight[1] * transfer_factor)
            for j, (copy, work, _) in s.active.items():
                steps.append(copy if copy else work / rates[j])
            if not steps or min(steps) <= 0:
                raise RuntimeError("Batch execution stalled")
            dt = min(steps)
            if len(work_ids) == 2:
                s.batch_ms += dt
            for j, (copy, work, kv) in tuple(s.active.items()):
                s.active[j] = (copy - dt, work, kv) if copy else (copy, work - dt * rates[j], kv)
            if s.flight is not None:
                obj, work = s.flight
                s.flight = obj, work - dt / transfer_factor
            s.now += dt
        return s
