"""Event controller for the R10 stationary, one-slot, shared-lane scope.

Wall clock, arrivals and actual operation completions belong to the driver.
This controller never estimates when a device operation is complete. Only
the frozen chooser uses the cost estimates to evaluate counterfactuals.
"""

from copy import copy
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol

from .live_snapshot import ObservedDecision, _check_scope, choose_observed
from .simulator import number


@dataclass(frozen=True)
class Read:
    request: int
    object_id: str
    lane_use: str


@dataclass(frozen=True)
class Compute:
    request: int
    boundary: int
    deps: tuple[str, ...]
    release: bool
    shared_checkpoint: bool


@dataclass(frozen=True)
class Launch:
    suffix_started: bool
    complete: bool = False


class Backend(Protocol):
    def reserve(self, request: int, deps: tuple[str, ...], active_kv: frozenset[str], now: Fraction) -> None: ...
    def read(self, action: Read, now: Fraction) -> None: ...
    def compute(self, action: Compute, now: Fraction) -> Launch: ...


class LiveFIFO:
    def __init__(self, exp, policy: str, backend: Backend, choose=None, observe=None):
        _check_scope(exp)
        self.exp = copy(exp)
        self.exp.arrivals = list(exp.arrivals)
        self.policy = policy
        self.backend = backend
        self.choose = choose or choose_observed
        self.observe = observe
        self.now = Fraction(0)
        self.head = 0
        self.arrived = 0
        self.chosen = []
        self.admitted = {}
        self.ready = set(exp.initial)
        self.reading = None
        self.read_at = None
        self.active = None
        self.compute_at = None
        self.suffix_at = None
        self.completions = []
        self.reservations = {}
        self.starts = []
        self.transfers = []
        self.peak_prefetch = 0
        self.ready_handoffs = []

    @property
    def done(self):
        return self.head == len(self.exp.requests)

    def _occupied(self):
        return set().union(*self.admitted.values())

    def snapshot(self, i):
        return ObservedDecision(self.now, self.head, self.arrived, i, tuple(self.chosen),
                                tuple(sorted(self.admitted.items())), frozenset(self.ready),
                                self.active.deps if self.active else (), self.compute_at, self.suffix_at,
                                self.reading.object_id if self.reading else None, self.read_at)

    def _commit(self, i):
        if i < len(self.chosen):
            return
        if i != len(self.chosen):
            raise RuntimeError('FIFO commit skipped a visible request')
        snapshot = self.snapshot(i)
        action = self.choose(self.exp, self.policy, snapshot)
        if action not in self.exp.eligible[i]:
            raise ValueError('Chooser returned an ineligible action')
        self.chosen.append(action)
        if self.observe is not None:
            self.observe(snapshot, action)

    def _deps(self, i):
        return self.exp.candidates[i][self.chosen[i]]

    def _complete(self):
        self.completions.append(self.now)
        self.head += 1
        self.active = self.compute_at = self.suffix_at = None

    def step(self, now, *, arrivals=(), read_done=None, suffix_started=None, compute_done=None):
        """Apply real observations at a clock tick, then issue eligible work.

        A batch must contain all observed ties. Arrival IDs are indexes in
        FIFO order. Read completion gives the active Read (identity-bearing).
        Compute/start completion gives the request index, not a prediction.
        """
        now = number(now)
        arrivals = tuple(arrivals)
        if now < self.now or self.done:
            raise ValueError('Non-monotonic or terminal executor event')
        if arrivals != tuple(range(self.arrived, self.arrived + len(arrivals))) or self.arrived + len(arrivals) > len(self.exp.requests):
            raise ValueError('Arrivals must be the next FIFO inputs')
        if read_done is not None and read_done != self.reading:
            raise ValueError('Completion from a non-current transfer')
        for request in (suffix_started, compute_done):
            if request is not None and (not self.active or request != self.active.request):
                raise ValueError('Completion/start from a non-current compute request')
        if suffix_started is not None and self.suffix_at is not None:
            raise ValueError('Duplicate suffix start observation')
        if compute_done is not None and self.suffix_at is None and suffix_started is None:
            raise ValueError('Compute completed before suffix start was observed')
        self.now = now
        if read_done is not None:
            self.ready.add(read_done.object_id)
            self.transfers.append((self.reading, self.read_at, now))
            self.reading = self.read_at = None
        if suffix_started is not None:
            self.suffix_at = now
        if compute_done is not None:
            self._complete()
        for i in arrivals:
            self.exp.arrivals[i] = now
        self.arrived += len(arrivals)
        self._issue()

    def _issue(self):
        while self.head < self.arrived:
            self._commit(self.head)
            deps = self._deps(self.head)
            if self.active is None and self.ready.issuperset(deps):
                released = self.admitted.pop(self.head, None) is not None
                occupied = self._occupied()
                shared = bool(deps and deps[-1] in (self.exp.initial | occupied) and
                              self.chosen[self.head] < self.exp.lengths[self.head])
                self.active = Compute(self.head, self.chosen[self.head], deps, released, shared)
                self.compute_at = self.now
                phase = self.backend.compute(self.active, self.now)
                if phase.complete and not phase.suffix_started:
                    raise RuntimeError('Backend completed compute without starting the suffix')
                self.suffix_at = self.now if phase.suffix_started else None
                self.starts.append((self.active, self.now))
                self.ready.intersection_update(occupied | self.exp.initial)
                if phase.complete:
                    self._complete()
                    continue  # original executor advances zero-duration work before reserving

            for i in range(self.head + 1, min(self.arrived, self.head + 1 + self.exp.window)):
                self._commit(i)
                future = tuple(o for o in self._deps(i) if o not in self.exp.initial)
                if i not in self.admitted and future and self.exp.size(self._occupied().union(future)) <= self.exp.budget:
                    borrowed = (frozenset(future).intersection(self.active.deps[:-1]) - self.ready
                                if self.active else frozenset())
                    self.backend.reserve(i, future, borrowed, self.now)
                    self.admitted[i] = future
                    self.ready.update(borrowed)
                    self.reservations[i] = self.now
                    if borrowed:
                        self.ready_handoffs.append((self.active.request, i, borrowed, self.now))

            self.peak_prefetch = max(self.peak_prefetch, self.exp.size(self._occupied()))
            if self.reading is None:
                sources = [(self.head, deps, 'demand')] if self.active is None else []
                sources.extend((i, self._deps(i), 'prefetch') for i in sorted(self.admitted))
                transfer = next((Read(i, o, mode) for i, objects, mode in sources for o in objects if o not in self.ready), None)
                if transfer is not None:
                    self.backend.read(transfer, self.now)
                    self.reading = transfer
                    self.read_at = self.now
            return
