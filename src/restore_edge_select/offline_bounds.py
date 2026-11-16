"""Certified branch and bound for fixed-order, single-transfer trace replay."""

from heapq import heappop, heappush
from itertools import count

from .simulator import OfflineBranch


def search(exp, feasible, max_nodes):
    """Return bounds on the minimum sum of responses, using exact clock arithmetic.

    feasible is a list of zero-decision-overhead simulator results. Each node
    fixes an initial sequence of decisions. The frontier covers every plan
    not already replayed or bounded by the incumbent.
    """
    if (exp.per_tier or exp.host_cache_bytes or exp._actual_transfer_times != exp._transfer_times
            or any(speeds != (1,) for _, speeds in exp.io_epochs.values())):
        raise ValueError('Oracle bound requires stationary costs and one transfer lane without host retention')
    incumbent = min(feasible, key=lambda r: r['response_sum_ms'])
    best, plan = incumbent['response_sum_ms'], tuple(incumbent['boundaries_tokens'])
    arrivals = exp.arrivals
    n, arrival_sum = len(arrivals), sum(arrivals)
    releases = {}
    deps = []
    for i, choices in enumerate(exp.eligible):
        row = {c: frozenset(exp.candidates[i][c]) - exp.initial for c in choices}
        deps.append(row)
        for objects in row.values():
            for obj in objects:
                releases.setdefault(obj, arrivals[i])
    io = exp._transfer_times
    # Even with perfect retention and no unrelated transfers, each required
    # object must first be loaded on the single lane after its earliest release.
    available = []
    for i, row in enumerate(deps):
        ends = {}
        for c, objects in row.items():
            end = 0
            for obj in sorted(objects, key=releases.__getitem__):
                end = max(end, releases[obj]) + io[obj]
            ends[c] = max(arrivals[i], end)
        available.append(ends)

    def floor(prefix, now, head, completions, ready, running, cf, transfer, tf, active):
        end, total = now, sum(completions)
        if running is not None:
            end += (running - now) / cf
            total += end
        free = ready | (set(active[:-1]) if running is not None else set())
        flight = transfer[1] if transfer else None
        flight_work = (transfer[0] - now) / tf if transfer else 0
        for i in range(head + (running is not None), n):
            choices = (prefix[i],) if i < len(prefix) else exp.eligible[i]
            def earliest(c):
                missing = deps[i][c] - free
                # No new transfer may overlap work already in flight. If the
                # in-flight object is itself needed, count its remaining work once.
                data = now + (flight_work + sum(io[obj] for obj in missing if obj != flight) if missing else 0)
                return max(end, available[i][c], data) + exp._suffix_times[i][c]
            end = min(map(earliest, choices))
            total += end
        return total - arrival_sum

    heap, serial = [], count()
    heappush(heap, (floor((), 0, 0, (), set(exp.initial), None, 1, None, 1, ()), 0, next(serial), ()))
    nodes = 0
    while heap and heap[0][0] < best and nodes < max_nodes:
        _, _, _, prefix = heappop(heap)
        nodes += 1

        def probe(i, now, head, completions, ready, running, cf, transfer, tf, active):
            assert i >= len(prefix) and all(len(c) == 1 for c in exp.eligible[len(prefix):i])
            prior = prefix + tuple(c[0] for c in exp.eligible[len(prefix):i])
            return [(floor((*prior, c), now, head, completions, ready, running, cf, transfer, tf, active),
                     (*prior, c)) for c in exp.eligible[i]]

        try:
            run = exp._run('fixed', 0, prefix + (None,) * (n - len(prefix)), 0, n,
                           offline_probe=probe, lean_rollout=True)
        except OfflineBranch as branch:
            for lb, child in branch.args[0]:
                if lb < best:
                    heappush(heap, (lb, -len(child), next(serial), child))
        else:
            if run['response_sum_ms'] < best:
                best, plan = run['response_sum_ms'], tuple(run['boundaries_tokens'])
    return min(best, heap[0][0]) if heap else best, best, plan, nodes
