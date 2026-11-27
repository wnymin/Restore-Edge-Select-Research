"""Replay original-time Weka parent/subagent overlaps with a resident anchor per prefix."""

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path

from scripts.data.audit_r11_tracelab import quantiles
from scripts.data.study_r11_weka import (BLOCK, POLICIES, PREFETCH_MIB, ReferenceTailExperiment, lcp, read,
                                         replay, terminal)


MIB = 2**20
FIXED_MIB = 4096
DEVICE_MIB = 8192
EXECUTION_MIB = 512
WORKSPACE_MIB = 128


def tree_rows(node, branch=()):
    for index, row in enumerate(node['requests']):
        if isinstance(row.get('requests'), list):
            yield from tree_rows(row, (*branch, index))
        else:
            yield row, branch


def matching(catalog, blocks):
    return [blocks[:n] for n in sorted(catalog, reverse=True) if blocks[:n] in catalog[n]]


def completed(row):
    duration = row.get('api_time')
    return row['t'] + duration if duration is not None and duration >= 0 else None


def case(rows, prefixes, ends, start, focus, spec, cost, *, deepest_peer, followers, short_only):
    t0 = rows[start]['t']
    model = rows[focus]['model']
    catalog = defaultdict(set)
    for j in range(start):
        if rows[j]['model'] == model and ends[j] is not None and ends[j] <= t0 and prefixes[j]:
            catalog[len(prefixes[j])].add(prefixes[j])

    indices = [j for j in range(start, focus + 1) if rows[j]['model'] == model]
    if followers:
        peer_end = max(ends[j] for j in indices[:-1] if ends[j] is not None)
        subsequent = [j for j in range(focus + 1, len(rows))
                      if rows[j]['t'] < peer_end and rows[j]['model'] == model]
        indices.extend(subsequent[:2])
    objects, requests, mapping = {}, [], []
    absolute_prefixes = {}
    weights = spec['payload']

    def runs(pre):
        return [{'token_id': b, 'count': BLOCK} for b in pre]

    def path(anchor, later):
        deps = []
        # Prefix hashes are session-local, so the entire path and anchor disambiguate suffixes.
        anchor_key = sha256(repr(anchor).encode('ascii')).hexdigest()
        tail = later[len(anchor):]
        for end in range(1, len(tail) + 1):
            prefix = tail[:end]
            key = f'{anchor_key}:kv:{prefix}'
            objects.setdefault(key, {'id': key, 'kind': 'attention_kv', 'tier': 'RAM',
                                    'start_token_exclusive': (end - 1) * BLOCK,
                                    'end_token_inclusive': end * BLOCK, 'bytes': BLOCK * weights['attention_bytes_per_token'],
                                    'prefix_token_runs': runs(prefix)})
            absolute_prefixes[key] = later[:len(anchor) + end]
            deps.append(key)
        key = f'{anchor_key}:ck:{tail}'
        objects.setdefault(key, {'id': key, 'kind': 'recurrent_and_conv_checkpoint', 'tier': 'SSD',
                                 'boundary_tokens': len(tail) * BLOCK, 'bytes': weights['checkpoint_bytes'],
                                 'prefix_token_runs': runs(tail)})
        absolute_prefixes[key] = later
        return [*deps, key]

    anchors = set()
    for j in indices:
        blocks = prefixes[j]
        size = rows[j]['in']
        if size >= spec['context_tokens'] or len(blocks) != size // BLOCK:
            return None, 'context_or_missing_blocks'
        hits = [p for p in matching(catalog, blocks) if len(p) * BLOCK < size]
        if j == focus and (len(hits) < 2 or (short_only and len(hits[0]) - len(hits[1]) > 2)):
            return None, 'target_pair_unavailable_at_window_start'
        if not hits:
            return None, 'resident_anchor_unavailable_for_peer'
        anchor_index = 0 if deepest_peer and j != focus else min(1, len(hits) - 1)
        anchor = hits[anchor_index]
        anchors.add(anchor)
        offset = len(anchor) * BLOCK
        length = size - offset
        candidates = [{'boundary_tokens': 0, 'load_order': []}]
        if hits[0] != anchor:
            candidates.append({'boundary_tokens': (len(hits[0]) - len(anchor)) * BLOCK,
                               'load_order': path(anchor, hits[0])})
        if length * weights['attention_bytes_per_token'] + weights['checkpoint_bytes'] > (EXECUTION_MIB - WORKSPACE_MIB) * MIB:
            return None, 'peer_suffix_exceeds_execution_budget' if j != focus else 'target_suffix_exceeds_execution_budget'
        prompt = runs(blocks[len(anchor):])
        if size % BLOCK:
            prompt.append({'token_id': -1 - j, 'count': size % BLOCK})
        requests.append({'id': str(j), 'arrival_ms': str((rows[j]['t'] - t0) * 1000),
                         'input_token_runs': prompt, 'candidates': candidates})
        mapping.append({'index': j, 'resident_tokens': offset, 'input_tokens': size,
                        'arrival_ms': requests[-1]['arrival_ms'],
                        'absolute_candidates': [offset + c['boundary_tokens'] for c in candidates]})

    initial = sorted(key for key, p in absolute_prefixes.items() if (
        p in anchors if objects[key]['kind'] == 'recurrent_and_conv_checkpoint'
        else any(a[:len(p)] == p for a in anchors)))

    kv_blocks = 0
    previous = ()
    for anchor in sorted(anchors):
        kv_blocks += len(anchor) - lcp(previous, anchor)
        previous = anchor
    resident_bytes = (kv_blocks * BLOCK * weights['attention_bytes_per_token']
                      + len(anchors) * weights['checkpoint_bytes'])
    fixed = FIXED_MIB + Fraction(resident_bytes, MIB)
    if fixed + EXECUTION_MIB + PREFETCH_MIB > DEVICE_MIB:
        return None, 'resident_prefixes_exceed_hbm'
    # Initial simulator objects are aliases into the already-counted resident prefix union.
    aliases_mib = Fraction(sum(objects[key]['bytes'] for key in initial), MIB)
    trace = {'objects': list(objects.values()), 'requests': requests, 'cost': cost, 'units': {'bytes_per_mib': MIB},
             'scheduling': {'future_window_requests': 4, 'initial_device_object_ids': initial},
             'memory': {'device_mib': DEVICE_MIB, 'fixed_mib': fixed - aliases_mib, 'execution_mib': EXECUTION_MIB,
                        'workspace_mib': WORKSPACE_MIB, 'kv_block_tokens': BLOCK, 'extra_alignment_bytes': 0}}
    return (trace, mapping, len(anchors), resident_bytes), 'accepted'


def serial_control(trace, spec):
    requests = [{**r, 'arrival_ms': str(i * 60000)} for i, r in enumerate(trace['requests'])]
    control = replay({**trace, 'requests': requests}, spec)
    if any(w > 0 for p in POLICIES for w in control[p]['queue_wait_ms']):
        raise ValueError('serial control still has queue wait')
    return control


def calibrated_replay(trace, spec, mapping, calibration):
    exact = {(i, boundary - row['resident_tokens']): calibration[(row['input_tokens'], boundary)]
             for i, row in enumerate(mapping) for boundary in row['absolute_candidates']
             if (row['input_tokens'], boundary) in calibration}
    target = len(mapping) - 1
    if not all((target, b - mapping[target]['resident_tokens']) in exact for b in mapping[target]['absolute_candidates']):
        raise ValueError('target geometry absent from paired GPU data')

    class MeasuredTargetExperiment(ReferenceTailExperiment):
        def recompute(self, i, c):
            if (i, c) in exact:
                return exact[i, c]
            return super().recompute(i, c)

    return replay(trace, spec, experiment=MeasuredTargetExperiment, capture_visibility=True)


def compare(frames):
    result = {}
    for baseline, policy in zip(POLICIES, POLICIES[1:]):
        gross, paid = [], []
        changed, slower = 0, 0
        for frame in frames:
            base, tested = frame['out'][baseline], frame['out'][policy]
            changed += sum(x != y for x, y in zip(base['boundaries'], tested['boundaries'], strict=True))
            paired = [x - y for x, y in zip(base['response_ms'], tested['response_ms'], strict=True)]
            slower += sum(x < -1e-9 for x in paired)
            gross.extend(paired)
            paid.extend(x - y for x, y in zip(base['injected_response_ms'], tested['injected_response_ms'], strict=True))
        result[f'{policy}_vs_{baseline}'] = {'action_changes': changed, 'gross_total_ms': sum(gross),
                                             'with_chooser_total_ms': sum(paid), 'slower_requests': slower}
    return result


def stats(frames):
    return {'windows': len(frames), 'window_requests': sum(len(x['requests']) for x in frames),
            'sessions': len({f['row'] for f in frames}),
            'queued_requests': {p: sum(w > 0 for f in frames for w in f['out'][p]['queue_wait_ms']) for p in POLICIES},
            'responses_ms': {p: quantiles([t for f in frames for t in f['out'][p]['response_ms']]) for p in POLICIES},
            'chooser_ns': {p: quantiles([ns for f in frames for _, ns in f['out'][p]['timing']]) for p in POLICIES},
            'comparisons': compare(frames)}


def report(paths, spec, cost, *, deepest_peer=False, followers=False, short_only=True, calibration=None):
    counts, inputs, encounters, frames = Counter(), [], [], []
    for path in paths:
        entry, = read(path)['rows']
        root = entry['row']
        if entry['truncated_cells'] or root['block_size'] != BLOCK or root['hash_id_scope'] != 'local':
            raise ValueError(f'incomplete or incompatible dataset row: {path}')
        rows, branches = zip(*sorted(tree_rows(root), key=lambda x: x[0]['t']), strict=True)
        prefixes = terminal(rows)
        ends = [completed(row) for row in rows]
        source = {'row': entry['row_idx'], 'id': root['id'], 'sha256': sha256(path.read_bytes()).hexdigest(),
                  'requests': len(rows), 'file': path.name}
        inputs.append(source)
        catalog = defaultdict(lambda: defaultdict(set))
        active = []
        for i, (row, branch, blocks) in enumerate(zip(rows, branches, prefixes, strict=True)):
            model = row['model']
            for j in active:
                if ends[j] <= row['t'] and prefixes[j]:
                    catalog[rows[j]['model']][len(prefixes[j])].add(prefixes[j])
            active = [j for j in active if ends[j] > row['t']]
            peers = [j for j in active if rows[j]['model'] == model]
            cross = [j for j in peers if branches[j] != branch]
            hits = matching(catalog[model], blocks)
            short = len(hits) >= 2 and len(hits[0]) - len(hits[1]) <= 2
            counts.update(requests=1, nested_requests=bool(branch), same_model_overlap=bool(peers),
                          cross_branch_same_model_overlap=bool(cross), two_checkpoints=len(hits) >= 2,
                          short_pair=short, short_pair_same_model_overlap=short and bool(peers),
                          short_pair_cross_branch_same_model_overlap=short and bool(cross),
                          short_pair_cross_branch_two_or_more_model_peers=short and bool(cross) and len(peers) >= 2)
            if ends[i] is not None:
                active.append(i)
            if not (cross and (short if short_only else len(hits) >= 2)):
                continue
            start = min(peers)
            episode, status = case(rows, prefixes, ends, start, i, spec, cost,
                                   deepest_peer=deepest_peer, followers=followers, short_only=short_only)
            counts[status] += 1
            record = {'row': source['row'], 'index': i, 'branch': branch, 'peer_indices': peers,
                      'cross_branch_peers': [{'index': j, 'branch': branches[j]} for j in cross],
                      'arrival_s': str(row['t']), 'boundaries_at_arrival': [len(h) * BLOCK for h in hits[1::-1]],
                      'target_input_tokens': row['in'], 'reason': status}
            encounters.append(record)
            if episode is not None:
                trace, mapping, anchors, byte_count = episode
                frames.append({**record, 'requests': mapping, 'resident_anchors': anchors,
                               'resident_bytes': byte_count,
                               'resident_extra_dependencies': len(trace['scheduling']['initial_device_object_ids']),
                               'out': replay(trace, spec), 'serial_control': serial_control(trace, spec)})
                if calibration:
                    frames[-1]['calibrated_target'] = calibrated_replay(trace, spec, mapping, calibration)

    result = {'scope': 'within-session same-provider-model parent/subagent and distinct-subagent overlaps; publication '
                     'frozen at window start; separate resident anchors; relative R10 tail-cost proxy',
            'deepest_peer_anchor': deepest_peer, 'include_at_most_two_followers': followers,
            'target_gap_limit_tokens': 128 if short_only else None,
            'source_manifest': inputs, 'counts': counts, 'observed_targets': encounters, 'frames': frames,
            **stats(frames), 'serial_control_comparisons': compare([{'out': f['serial_control']} for f in frames]),
            'by_session_split': {'0-5': stats([f for f in frames if f['row'] < 6]),
                                                 '6-11': stats([f for f in frames if 6 <= f['row'] < 12]),
                                                 '12-and-later': stats([f for f in frames if f['row'] >= 12])}}
    if calibration:
        result['calibrated_target'] = stats([{**f, 'out': f['calibrated_target']} for f in frames])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('paths', nargs='+', type=Path)
    parser.add_argument('--deepest-peer-anchor', action='store_true')
    parser.add_argument('--followers', action='store_true')
    parser.add_argument('--any-gap', action='store_true')
    parser.add_argument('--target-calibration', type=Path, help='paired GPU medians for the main cohort targets')
    args = parser.parse_args()
    spec = read(Path('configs/qwen3.5-4b.json'))['models'][0]
    cost = read(Path('configs/r10-local-holdout-cost-v1.json'))['cost']
    if args.target_calibration and (args.deepest_peer_anchor or args.followers or args.any_gap):
        parser.error('target calibration applies to the main cohort')
    calibration = None
    if args.target_calibration:
        source = read(args.target_calibration)
        calibration = {(shape['receipt']['boundary'] + shape['receipt']['tail'], int(depth)): Fraction(value)
                       for shape in source['groups']['trajectory']
                       for depth, value in shape['prefill_median_ms_by_depth'].items()}
    result = report(args.paths, spec, cost, deepest_peer=args.deepest_peer_anchor, followers=args.followers,
                    short_only=not args.any_gap, calibration=calibration)
    if args.target_calibration:
        result['calibration_sha256'] = sha256(args.target_calibration.read_bytes()).hexdigest()
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: result[k] for k in ('counts', 'windows', 'window_requests', 'queued_requests', 'comparisons')}, indent=2))


if __name__ == '__main__':
    main()
