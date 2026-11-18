"""Compare GPU restore timing across logical batch caps at one physical cap."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
from statistics import mean, median

from scripts.gpu.experiments.study_dynamic_gpu_restore_queue import matched_trials


def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def number(row, name):
    return float(row[name])


def elapsed(row):
    target, peer, release = (number(row, k) for k in ('target_finish_ms', 'peer_finish_ms', 'release_ms'))
    return target-release, peer, target-release+peer, max(target, peer)


def compare(before, after):
    base, reduced = matched_trials(before, after)
    before_batch, = {int(r['batch_tokens']) for r in base.values()}
    after_batch, = {int(r['batch_tokens']) for r in reduced.values()}
    micro, = {int(r['ubatch_tokens']) for r in before + after}
    if not 0 < micro <= after_batch < before_batch or list(base) != list(reduced):
        raise ValueError('Expected a smaller logical cap, same physical cap, and matched trial order')
    if any(base[k]['greedy_token'] != reduced[k]['greedy_token'] for k in base):
        raise ValueError('First token differs across batch caps')
    groups = defaultdict(list)
    resident = defaultdict(dict)
    for table in (base, reduced):
        for r in table.values():
            if r['requested_lead_ms'] == '-1':
                resident[r['batch_tokens']][r['seed'], r['peer_total_tokens'], r['rep']] = r
    for k, a in base.items():
        b = reduced[k]
        groups[int(a['peer_total_tokens']), int(a['requested_lead_ms'])].append((a, b))
    print(f'Logical {before_batch}->{after_batch}, physical {micro}, {len(base)} matches with equal first token')
    for (peer, lead), pairs in sorted(groups.items()):
        cols = list(zip(*(tuple(x-y for x,y in zip(elapsed(a), elapsed(b))) for a,b in pairs)))
        gains = ', '.join(f'{mean(c):+.2f}' for c in cols)
        print(f'{peer}/{lead} ({len(pairs)}): smaller cap target, peer, sum, makespan gain {gains} ms')
        if lead < 0:
            continue
        for k in ('restore_queue_ms', 'restore_dispatch_from_peer_post_ms', 'restore_dispatch_from_peer_release_ms',
                  'restore_server_ms_trace', 'restore_before_metadata_ms', 'restore_metadata_to_result_ms',
                  'first_batch_gap_ms', 'target_batch_to_release_ms', 'peer_release_from_target_batch_ms'):
            meds = [median(number(pair[i], k) for pair in pairs) for i in (0, 1)]
            delta = [number(a, k)-number(b, k) for a,b in pairs]
            print(f'  {k}: {meds[0]:.2f} -> {meds[1]:.2f} ms, paired gain median {median(delta):+.2f}; '
                  f'earlier in {sum(d > 0 for d in delta)}/{len(delta)}')
        for i, cap in enumerate((before_batch, after_batch)):
            arms = [p[i] for p in pairs]
            ready = sum(number(r, 'restore_end_ms') <= number(r, 'release_ms') for r in arms)
            earlier = sum(number(r, 'restore_dispatch_from_peer_release_ms') < 0 for r in arms)
            effects = [tuple(x-y for x,y in zip(elapsed(resident[str(cap)][r['seed'], r['peer_total_tokens'], r['rep']]),
                                                 elapsed(r))) for r in arms]
            gain_str = ', '.join(f'{mean(c):+.2f}' for c in zip(*effects))
            print(f'  cap {cap}: ready {ready}/{len(arms)}, dispatched while peer active {earlier}/{len(arms)}; '
                  f'deep vs resident target, peer, sum, makespan gain {gain_str} ms; '
                  f'target faster in {sum(e[0] > 0 for e in effects)}/{len(arms)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('larger', type=Path)
    parser.add_argument('smaller', type=Path)
    args = parser.parse_args()
    compare(read(args.larger), read(args.smaller))


if __name__ == '__main__':
    main()
