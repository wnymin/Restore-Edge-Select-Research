"""Check a two-request boundary model and summarize the value of timing information."""

from collections import Counter, defaultdict
from itertools import product


# Each candidate is (staging footprint, transfer duration, continuation duration).
SHALLOW = (1, 1, 4)


def schedule(candidate, capacity, window, arrival, load, release):
    """Evaluate one complete path in FIFO; target transfers begin at time zero."""
    size, ready, remaining = candidate
    start = max(window, ready)
    finish = start + remaining
    free = start if release == 'handoff' else finish
    # Peer footprint is 2; all candidates in this study fit individually.
    admitted = arrival if size + 2 <= capacity else max(arrival, free)
    peer_finish = max(finish, admitted + load) + 1
    return finish, peer_finish


def margin(shallow, deep, capacity, window, arrival, load, release):
    """Predict improvement in response sum by going deeper, from model inputs."""
    ends, exposed = [], []
    for size, ready, remaining in (shallow, deep):
        start = max(window, ready)
        end = start + remaining
        free = start if release == 'handoff' else end
        wait = max(0, free - arrival) if size + 2 > capacity else 0
        ends.append(end)
        exposed.append(max(0, arrival + load + wait - end))
    return 2 * (ends[0] - ends[1]) + exposed[0] - exposed[1]


def main():
    examples = [(3, 2, (10, 11), (8, 15)), (3, 10, (10, 19), (8, 19)),
                (4, 2, (10, 11), (8, 11))]
    for cap, arrive, s, d in examples:
        assert schedule(SHALLOW, cap, 6, arrive, 8, 'handoff') == s
        assert schedule((2, 2, 2), cap, 6, arrive, 8, 'handoff') == d

    shallow = schedule(SHALLOW, 3, 6, 2, 6, 'handoff')
    deep = schedule((2, 2, 1), 3, 6, 2, 6, 'handoff')
    assert (shallow, deep) == ((10, 11), (7, 13))
    assert sum(shallow) - sum(deep) == 1  # First peer alone does not reverse the choice.
    assert sum(shallow) + 12 - (sum(deep) + 14) == -1  # A resident third request does.
    for base, delay, ready in product(range(12), range(12), range(24)):
        assert max(base + delay, ready) - max(base, ready) == max(0, delay - max(0, ready - base))

    grid = product(('handoff', 'completion'), (2, 3, 4), (0, 1, 2, 4, 6),
                   (1, 2, 3, 4), (1, 2, 3, 4), range(2, 13), range(1, 10))
    stats = defaultdict(Counter)
    for release, cap, window, ready, remaining, arrive, load in grid:
        if arrive < ready:
            continue   # Isolate staging: no conflict with the target's own transfer.
        deep = (2, ready, remaining)
        s, d = [schedule(c, cap, window, arrive, load, release) for c in (SHALLOW, deep)]
        scores = [sum(s), sum(d)]  # Arrival times are constant across arms.
        gap = scores[0] - scores[1]
        predicted = margin(SHALLOW, deep, cap, window, arrive, load, release)
        assert predicted == gap
        scores = [v - min(scores) for v in scores]
        row = stats[release, cap]
        row['rows'] += 1
        row['local strictly deep'] += d[0] < s[0]
        row['local strictly deep, global shallow'] += d[0] < s[0] and gap < 0
        for label, select_deep in [('longest', True), ('static', ready + remaining < 5),
                                   ('target only', d[0] < s[0]), ('peer known', predicted > 0)]:
            error = scores[select_deep]
            row[label + ' errors'] += error > 0
            row[label + ' regret'] += error

    policies = ['longest', 'static', 'target only', 'peer known']
    print('release cap rows strict_deep strict_reversal ' + ' '.join(f'{p}(errors/total_regret)' for p in policies))
    for (release, cap), row in sorted(stats.items()):
        print(release, cap, row['rows'], row['local strictly deep'], row['local strictly deep, global shallow'],
              *(f"{row[p+' errors']}/{row[p+' regret']}" for p in policies))

    print('release strict_local_deep certified_deep ambiguous (capacity 3; future arrival unknown)')
    for release in ('handoff', 'completion'):
        strict, certified = 0, 0
        for window, ready, remaining, load in product((0, 1, 2, 4, 6), range(1, 5), range(1, 5), range(1, 10)):
            deep = (2, ready, remaining)
            earliest = max(ready, 2)
            gain = max(window, 1) + 4 - (max(window, ready) + remaining)
            if gain <= 0:
                continue
            strict += 1
            worst = min(gain, margin(SHALLOW, deep, 3, window, earliest, load, release))
            # Beyond both releases and completions all later arrivals give exactly gain.
            sampled = [margin(SHALLOW, deep, 3, window, a, load, release) for a in range(earliest, 16)]
            assert min(sampled) == worst
            certified += worst >= 0
        print(release, strict, certified, strict - certified)


if __name__ == '__main__':
    main()
