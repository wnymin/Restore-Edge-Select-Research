"""Audit Weka prefix geometry and replay retained-anchor episodes using the R10 simulator."""

import argparse
from bisect import insort
from collections import Counter, defaultdict
from decimal import Decimal
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

from restore_edge_select.live_snapshot import _options, audit_simulated_decisions, choose_observed
from restore_edge_select.prefill_cost import sampled_cost
from restore_edge_select.simulator import Experiment
from scripts.data.audit_r11_tracelab import quantiles


BLOCK = 64
HISTORY = 8
ONLINE = 4
SUFFIX_LIMIT = 2560
PILOT_SUFFIX_LIMIT = 16384
PREFETCH_MIB = Fraction('82.25')
POLICIES = ('longest', 'local-cost', 'entry-replay')


def read(path):
    return json.loads(path.read_bytes(), parse_float=Decimal)


def flatten(trace):
    records = []
    for row in trace['requests']:
        if isinstance(row.get('requests'), list):
            records.extend(flatten(row))
        else:
            records.append(row)
    return records


def lcp(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def terminal(records):
    """Use complete prompt blocks and an observed historical input endpoint."""
    return [tuple(row['hash_ids'][:row['in'] // BLOCK]) for row in records]


def geometry(rows, prefixes):
    counts = Counter()
    samples = defaultdict(list)
    admitted = defaultdict(lambda: defaultdict(set))
    depths = defaultdict(list)
    finished = []
    missing_timing = 0
    previous = ()
    previous_row = None
    overlap_cases = []
    for index, (row, blocks) in enumerate(zip(rows, prefixes, strict=True)):
        now = row['t']
        model = row['model']
        for when, earlier_model, earlier in finished:
            if when <= now and earlier:
                depth = len(earlier)
                catalog = admitted[earlier_model]
                if depth not in catalog:
                    insort(depths[earlier_model], depth)
                catalog[depth].add(earlier)
        finished = [(when, m, p) for when, m, p in finished if when > now]
        overlap = bool(finished)
        busy = [when - now for when, m, _ in finished if m == model]
        same_model_overlap = bool(busy)
        counts['with_provider_overlap'] += overlap
        counts['with_same_model_overlap'] += same_model_overlap
        observed = row.get('api_time')
        if observed is not None and observed >= 0:
            finished.append((now + observed, model, blocks))
        else:
            missing_timing += 1
        hits = []
        for depth in reversed(depths[model]):
            if depth <= len(blocks) and blocks[:depth] in admitted[model][depth]:
                hits.append(depth)
                if len(hits) == 2:
                    break
        stop = previous_row.get('stop') if previous_row else None
        group = {None: 'tool_status_unknown', 'tool_use': 'after_tool_use'}.get(stop, 'after_other_stop')
        counts.update(requests=1, exact_checkpoint=bool(hits), two_checkpoints=len(hits) == 2)
        counts[f'{group}_requests'] += 1
        if len(hits) == 2:
            gap = (hits[0] - hits[1]) * BLOCK
            counts[f'{group}_two_checkpoints'] += 1
            counts[f'{group}_gap_le128'] += gap <= 128
            counts['gap_le128'] += gap <= 128
            counts['gap_le128_with_provider_overlap'] += gap <= 128 and overlap
            counts['gap_le128_with_same_model_overlap'] += gap <= 128 and same_model_overlap
            if gap <= 128 and same_model_overlap:
                overlap_cases.append({'request_index': index, 'arrival_seconds': str(now),
                                      'input_tokens': row['in'], 'boundaries': [h * BLOCK for h in reversed(hits)],
                                      'provider_peers_still_active': len(busy),
                                      'provider_nearest_peer_completion_ms': float(1000 * min(busy))})
            samples['nearest_two_gap_tokens'].append(gap)
        samples['input_tokens'].append(row['in'])
        if previous_row is not None:
            samples['inter_request_ms'].append(float(1000 * (now - previous_row['t'])))
            if previous_row['model'] == model:
                samples['mutation_blocks_previous'].append(len(previous) - lcp(blocks, previous))
        previous = blocks
        previous_row = row
    counts['missing_api_time'] = missing_timing
    return counts, samples, overlap_cases


def make_case(rows, prefixes, spec, cost):
    """Freeze history before replay. Boundary zero reuses the common warm anchor."""
    warm, online = rows[:HISTORY], rows[HISTORY:]
    past, requests = prefixes[:HISTORY], prefixes[HISTORY:]
    time_zero = online[0]['t']
    model = online[0]['model']
    if any(row['model'] != model for row in online):
        return None, 'mixed_provider_models'
    histories = [p for row, p in zip(warm, past, strict=True)
                 if row['model'] == model and row.get('api_time') is not None and row['api_time'] >= 0
                 and row['t'] + row['api_time'] <= time_zero]
    if len(histories) < 2:
        return None, 'insufficient_completed_history'
    # Choose the third-latest distinct known endpoint (or the earliest of two),
    # before inspecting any of the replay inputs.
    distinct = list(dict.fromkeys(reversed(histories)))
    if len(distinct) < 2:
        return None, 'insufficient_distinct_history'
    anchor = distinct[min(2, len(distinct) - 1)]
    if not anchor or any(p[:len(anchor)] != anchor for p in requests):
        return None, 'common_anchor_absent'
    n = len(anchor)
    lengths = [r['in'] - BLOCK * n for r in online]
    if any(length <= 0 or length > PILOT_SUFFIX_LIMIT for length in lengths):
        return None, 'suffix_outside_pilot_limit'
    if any(len(p) != r['in'] // BLOCK for p, r in zip(requests, online, strict=True)):
        return None, 'missing_complete_input_hash_blocks'
    catalog = sorted({p[n:] for p in histories if p[:n] == anchor and 0 < len(p) - n <= PILOT_SUFFIX_LIMIT // BLOCK},
                     key=lambda p: (len(p), p))
    if not catalog:
        return None, 'no_post_anchor_checkpoint'
    shallow = catalog[0]
    objects = {}

    def token_runs(blocks):
        return [{'token_id': value, 'count': BLOCK} for value in blocks]

    def add_objects(blocks):
        deps = []
        for end in range(1, len(blocks) + 1):
            pre = blocks[:end]
            identifier = 'kv:' + ':'.join(map(str, pre))
            objects.setdefault(identifier, {'id': identifier, 'kind': 'attention_kv', 'tier': 'RAM',
                                            'start_token_exclusive': (end - 1) * BLOCK,
                                            'end_token_inclusive': end * BLOCK,
                                            'bytes': BLOCK * spec['payload']['attention_bytes_per_token'],
                                            'prefix_token_runs': token_runs(pre)})
            deps.append(identifier)
        identifier = 'ck:' + ':'.join(map(str, blocks))
        objects.setdefault(identifier, {'id': identifier, 'kind': 'recurrent_and_conv_checkpoint',
                                        'tier': 'RAM' if blocks == shallow else 'SSD',
                                        'boundary_tokens': len(blocks) * BLOCK,
                                        'bytes': spec['payload']['checkpoint_bytes'],
                                        'prefix_token_runs': token_runs(blocks)})
        return [*deps, identifier]

    simulated = []
    for i, (r, blocks, length) in enumerate(zip(online, requests, lengths, strict=True)):
        post_anchor = blocks[n:]
        runs = token_runs(post_anchor)
        if length % BLOCK:
            runs.append({'token_id': -1 - i, 'count': length % BLOCK})
        candidates = [{'boundary_tokens': 0, 'load_order': []}]
        for checkpoint in catalog:
            if post_anchor[:len(checkpoint)] == checkpoint:
                candidates.append({'boundary_tokens': len(checkpoint) * BLOCK, 'load_order': add_objects(checkpoint)})
        simulated.append({'id': str(i), 'arrival_ms': str((r['t'] - time_zero) * 1000),
                          'input_token_runs': runs, 'candidates': candidates})
    if all(len(r['candidates']) == 1 for r in simulated):
        return None, 'no_post_anchor_hit'
    resident = add_objects(shallow)
    fixed_mib = Fraction(4096) + Fraction(n * BLOCK * spec['payload']['attention_bytes_per_token']
                                               + spec['payload']['checkpoint_bytes'], 2**20)
    trace = {'objects': list(objects.values()), 'requests': simulated, 'cost': cost, 'units': {'bytes_per_mib': 2**20},
             'scheduling': {'future_window_requests': 4, 'initial_device_object_ids': resident},
             'memory': {'device_mib': 8192, 'fixed_mib': fixed_mib, 'execution_mib': 512, 'workspace_mib': 128,
                        'kv_block_tokens': BLOCK, 'extra_alignment_bytes': 0}}
    if fixed_mib + 512 + PREFETCH_MIB + Fraction(sum(objects[k]['bytes'] for k in resident), 2**20) > 8192:
        return None, 'anchor_exceeds_hbm'
    if max(lengths) * spec['payload']['attention_bytes_per_token'] + spec['payload']['checkpoint_bytes'] > 384 * 2**20:
        return None, 'suffix_exceeds_execution_budget'
    return trace, 'accepted_extrapolated_suffix' if max(lengths) > SUFFIX_LIMIT else 'accepted_short_suffix'


class ReferenceTailExperiment(Experiment):
    """Use the final measured cached-position column for all longer offset shapes."""

    def __init__(self, trace, spec):
        model = trace['cost']['recompute']
        self.chunk = model['chunk_tokens']
        anchor = max(row['cached_tokens'] for row in model['samples'])
        if anchor != SUFFIX_LIMIT:
            raise ValueError('unexpected R10 calibration grid')
        per_chunk = lru_cache(maxsize=4096)(sampled_cost(model))
        self.tail_cost = lambda at, size: per_chunk(min(at, anchor), size)
        super().__init__(trace, spec, PREFETCH_MIB)

    def recompute(self, i, c):
        suffix = self.lengths[i] - c
        return sum((self.tail_cost(c + at, min(self.chunk, suffix - at))
                    for at in range(0, suffix, self.chunk)), Fraction(0))


def replay(trace, spec, *, experiment=ReferenceTailExperiment, capture_visibility=False):
    exp = experiment(trace, spec)
    out = {}
    for policy in POLICIES:
        mode, options = policy, {}
        if policy == 'entry-replay':
            mode, frozen = _options(policy)
            options = {k: frozen[k] for k in ('score_candidate', 'screen_mode', 'queue_horizon')}
            options['screen_search'] = 'entry-replay'
        result = exp.run(mode, profile_decisions=True, **options)
        visibility = []
        if policy == 'entry-replay':
            def verify(observation, selected):
                if choose_observed(exp, policy, observation) != selected:
                    raise ValueError('snapshot chooser disagrees with closed-loop R11 replay')
                if capture_visibility:
                    visibility.append({'request': observation.request,
                                       'earlier_unfinished': observation.request - observation.head,
                                       'later_arrived': observation.arrived - observation.request - 1,
                                       'io_active': observation.transfer_object is not None})

            verified = audit_simulated_decisions(exp, policy, verify)
            if result['boundaries_tokens'] != verified['boundaries_tokens']:
                raise ValueError('closed-loop pilot differs from the frozen R10 chooser')
        vals = [float(end - start) for end, start in zip(result['completion_ms'], exp.arrivals, strict=True)]
        paid = exp.run(mode, inject_measured=True, **options)
        out[policy] = {'response_ms': vals, 'queue_wait_ms': [float(t) for t in result['queue_wait_ms']],
                       'injected_response_ms': [float(end - start)
                       for end, start in zip(paid['completion_ms'], exp.arrivals, strict=True)],
                       'boundaries': result['boundaries_tokens'], 'timing': result['decision_profile_ns']}
        if policy == 'entry-replay' and capture_visibility:
            out[policy]['visibility'] = visibility
    return out


def report(sources, spec, cost):
    release_counts = defaultdict(Counter)
    distributions = defaultdict(lambda: defaultdict(list))
    frames = []
    manifest = []
    overlap_cases = []
    for path in sources:
        raw = read(path)
        entry, = raw['rows']
        if entry['truncated_cells']:
            raise ValueError(f'truncated dataset viewer row: {path}')
        trace = entry['row']
        if trace['block_size'] != BLOCK or trace['hash_id_scope'] != 'local':
            raise ValueError(f'unexpected hash semantics: {path}')
        release = path.name.split('-')[0]
        rows = sorted(flatten(trace), key=lambda r: r['t'])
        key = f'{release}/{trace["id"]}'
        manifest.append({'source': path.name, 'row_index': entry['row_idx'], 'trace': key,
                         'requests': len(rows), 'sha256': sha256(path.read_bytes()).hexdigest()})
        prefixes = terminal(rows)
        counts, samples, observed_cases = geometry(rows, prefixes)
        overlap_cases.extend({'trace': key, **case} for case in observed_cases)
        release_counts[release].update(counts)
        release_counts[release]['traces'] += 1
        for name, values in samples.items():
            distributions[release][name].extend(values)
        for i in range(HISTORY, len(rows) - ONLINE + 1, ONLINE):
            trace_case, reason = make_case(rows[i-HISTORY:i+ONLINE], prefixes[i-HISTORY:i+ONLINE], spec, cost)
            release_counts[release][f'frame_{reason}'] += 1
            if trace_case is not None:
                frames.append({'trace': key, 'offset': i, 'scope': reason, 'out': replay(trace_case, spec)})
    responses = {p: [v for frame in frames for v in frame['out'][p]['response_ms']] for p in POLICIES}
    timings = {p: [ns for frame in frames for _, ns in frame['out'][p]['timing']] for p in POLICIES}
    queued = {p: sum(wait > 0 for frame in frames for wait in frame['out'][p]['queue_wait_ms']) for p in POLICIES}
    comparisons = {}
    for baseline, policy in zip(POLICIES, POLICIES[1:]):
        actions = response_delta = injected_delta = slower = 0
        for frame in frames:
            base, tested = frame['out'][baseline], frame['out'][policy]
            actions += sum(a != b for a, b in zip(base['boundaries'], tested['boundaries'], strict=True))
            response_delta += sum(base['response_ms']) - sum(tested['response_ms'])
            injected_delta += sum(base['injected_response_ms']) - sum(tested['injected_response_ms'])
            slower += sum(t > b for t, b in zip(tested['response_ms'], base['response_ms'], strict=True))
        comparisons[f'{policy}_vs_{baseline}'] = {
            'action_differences': actions, 'gross_saving_total_ms': response_delta,
            'injected_saving_total_ms': injected_delta, 'slower_requests_excluding_overhead': slower}
    return {'scope': 'first-pass retained-anchor sensitivity, full input context cost excluded',
            'history': HISTORY, 'online': ONLINE, 'prefetch_mib': float(PREFETCH_MIB),
            'source_manifest': manifest, 'counts': release_counts,
            'short_gap_provider_overlap_cases': overlap_cases,
            'distributions': {release: {k: {'n': len(v), **quantiles(v)} for k, v in data.items()}
                              for release, data in distributions.items()},
            'pilot': {'requests': len(responses['local-cost']), 'queue_wait_requests': queued,
                      'response_ms': {p: quantiles(v) for p, v in responses.items()},
                      'chooser_ns': {p: quantiles(v) for p, v in timings.items()}, 'comparisons': comparisons},
            'frames': frames}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('paths', nargs='+', type=Path)
    args = parser.parse_args()
    spec = read(Path('configs/qwen3.5-4b.json'))['models'][0]
    cost = read(Path('configs/r10-local-holdout-cost-v1.json'))['cost']
    args.out.write_text(json.dumps(report(sorted(args.paths), spec, cost), indent=2) + '\n',
                        encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
