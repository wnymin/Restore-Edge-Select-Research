"""Characterize production-derived inputs and replay margins in the R10 holdouts."""

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

from restore_edge_select.live_snapshot import audit_simulated_decisions, choose_observed
from restore_edge_select.simulator import Experiment
from scripts.data.build_r10_bailian import BLOCK, DEPTHS, HISTORY, PROMPT_MAX, WINDOW, past_windows, row_blocks, sha, windows
from scripts.gpu.analysis.diagnose_r10_boundary_agreement import (
    InstrumentedExperiment, archive_records, expose, reconstruct, serial_components, summarize,
)
from scripts.gpu.analysis.explain_r10_controlled_choices import scaled


ARRIVAL_SCALES = ('4', '1', '1/4', '1/16')
PREFILL_SCALES = ('1', '2', '4')
SSD_SCALES = ('8', '16')


def read(path):
    return json.loads(path.read_bytes())


def describe(samples):
    return {name: summarize(series) for name, series in samples.items() if series}


def source_window(lines, counts, samples):
    rows = [json.loads(line, parse_float=Decimal) for line in lines]
    times = [row['timestamp'] * 1000 for row in rows]
    if any(a > b for a, b in zip(times, times[1:])):
        raise ValueError('out-of-order source arrivals')
    counts['windows'] += 1
    seen, historical = {}, set()
    depths = Counter()
    for i, row in enumerate(rows):
        _, hashes = row_blocks(row)
        keys = [hashes[:d // BLOCK] for d in DEPTHS if len(hashes) >= d // BLOCK]
        if i < HISTORY:
            historical.update(keys)
        else:
            available = [k for k in keys if k in historical]
            depth = len(available[-1]) * BLOCK if available else 0
            depths[depth] += 1
            counts.update(dict(requests=1, with_checkpoint=bool(available), multiple_checkpoints=len(available) > 1,
                               input_over_limit=row['input_length'] > PROMPT_MAX))
            samples['input_tokens'].append(row['input_length'])
            samples['output_tokens'].append(row['output_length'])
            if i > HISTORY:
                samples['arrival_gap_ms'].append(times[i] - times[i - 1])
            if available:
                prior = seen[available[-1]]
                elapsed = times[i] - times[prior]
                samples['deepest_match_lag_requests'].append(i - prior)
                samples['deepest_match_lag_ms'].append(elapsed)
                counts['deepest_match_in_last_4_requests'] += i - prior <= 4
                counts['deepest_match_in_last_250ms'] += elapsed <= 250
                counts['deepest_match_in_last_1000ms'] += elapsed <= 1000
            if i + 1 < len(rows):
                near = times[i + 1] - times[i] <= 250
                counts.update(dict(next_gap_observable=1, next_arrival_within_250ms=near,
                                   multiple_and_next_gap_observable=len(available) > 1,
                                   multiple_and_next_arrival_within_250ms=len(available) > 1 and near))
        seen.update((key, i) for key in keys)
    counts.update({f'deepest_{depth}': count for depth, count in depths.items()})
    return depths


def source_study(raw, evidence):
    manifest = read(Path('data/bailian-manifest.json'))
    earlier = past_windows()
    registered = {(r['source'], r['window']): r for r in read(evidence / 'r10-bailian-inputs.json')['cases']}
    held_out = set()
    for phase in ('', 'service-'):
        for block in read(evidence / f'r10-local-{phase}holdout-execution-lock.json')['blocks']:
            if block['family'] == 'bailian':
                key = block['source'], block['raw_rows'][0] // WINDOW
                case = registered[key]
                if (any(block[k] != case[k] for k in ('raw_rows', 'raw_window_sha256'))
                        or block['source_sha256'] != case['generated_sha256']):
                    raise ValueError(f'changed selection identity: {key}')
                held_out.add(key)
    results = {}
    verified = set()
    for source in manifest['files']:
        name = source['name']
        path = raw / name
        if path.stat().st_size != source['bytes'] or sha(path) != source['sha256']:
            raise ValueError(f'changed raw source: {name}')
        by_split = {'r10_holdout': (Counter(), defaultdict(list)), 'outside_registered': (Counter(), defaultdict(list))}
        skipped = 0
        for index, lines in enumerate(windows(path)):
            key = name, index
            if key in held_out:
                r = registered[key]
                if (sha256(b''.join(lines)).hexdigest() != r['raw_window_sha256']
                        or r['source_sha256'] != source['sha256']):
                    raise ValueError(f'changed holdout window: {key}')
                depths = source_window(lines, *by_split['r10_holdout'])
                if (sum(v for d, v in depths.items() if d), sum(v for d, v in depths.items() if d >= DEPTHS[1])) != (
                        r['with_checkpoints'], r['with_multiple_checkpoints']):
                    raise ValueError(f'candidate coverage differs from builder: {key}')
                verified.add(key)
            elif key in registered or key in earlier:
                skipped += 1
            else:
                source_window(lines, *by_split['outside_registered'])
        results[name] = dict(source_sha256=source['sha256'], skipped_registered_windows=skipped,
                             **{split: dict(counts=dict(counts), distribution=describe(samples))
                                for split, (counts, samples) in by_split.items()})
    if verified != held_out:
        raise ValueError('incomplete holdout windows')
    return results


def compare_window(exp, obs, base, choices, references, compute, margins):
    i = obs.request
    index = i - obs.head
    ends = exp.forecasts[base]
    weight = len(ends) - index
    if exp.forecasts.keys() != set(choices) or len(ends) != min(obs.arrived, i + 5) - obs.head:
        raise ValueError('candidate forecast population differs')
    for alt in sorted(set(choices) - {base}):
        other = exp.forecasts[alt]
        delta = [a - b for a, b in zip(other, ends, strict=True)]
        # Other boundaries are fixed. Only the target's nominal compute work changes.
        saving = compute[alt] - compute[base]
        previous = [0, *delta[:-1]]
        gap_deltas = [value - last for value, last in zip(delta, previous, strict=True)]
        gap_deltas[index] -= saving
        relief = -sum((len(ends) - j) * gap for j, gap in enumerate(gap_deltas))
        penalty = weight * saving
        total = sum(other) - sum(ends)
        if saving <= 0 or penalty - relief != total:
            raise ValueError('invalid work decomposition')
        for key in ('all', 'to_recompute' if alt == 0 else 'to_shallower_checkpoint'):
            k = margins[key]
            k['gap_relief_fraction'].append(relief / penalty)
            k['weighted_compute_penalty_ms'].append(penalty)
            k['weighted_gap_relief_ms'].append(relief)
            k['window_delta_ms'].append(total)
            k['ahead_completion_delta_ms'].append(sum(delta[:index]))
            k['target_completion_delta_ms'].append(delta[index])
            k['follower_completion_delta_ms'].append(sum(delta[index+1:]))
        yield dict(**references, alternative=alt, target_weight=weight, window_delta_ms=total,
                   gap_relief_ms=relief, compute_penalty_ms=penalty, relief_fraction=relief / penalty,
                   ahead_gain=sum(delta[:index]) < 0, target_gain=delta[index] < 0,
                   followers_gain=sum(delta[index+1:]) < 0)


def note_snapshot(counts, exp, obs, choices, base):
    n = len(choices) - 1
    i = obs.request
    d = exp.features.get(base, {})
    temporary_block = any(c and i > obs.head and not row['prefetch_fits'] and row['extra_reserve_bytes'] <= exp.budget
                          for c, row in exp.features.items())
    admitted_shallower = any(0 < c < base and row['prefetch_fits'] for c, row in exp.features.items())
    fit_asymmetry = admitted_shallower and not d['prefetch_fits']
    counts.update(dict(snapshots=1, checkpoint=n > 0, multiple_checkpoints=n > 1,
                       with_checkpoint_ahead=n > 0 and i > obs.head,
                       with_checkpoint_followers=n > 0 and obs.arrived > i + 1,
                       with_checkpoint_active=n > 0 and obs.compute_start_ms is not None,
                       with_checkpoint_active_zero_residual=n > 0 and obs.compute_start_ms is not None and d['head_slack_ms'] == 0,
                       with_checkpoint_inflight=n > 0 and obs.transfer_object is not None,
                       with_checkpoint_reserved_overlap=bool(d.get('shared_reservation_bytes')),
                       multiple_and_inflight=n > 1 and obs.transfer_object is not None,
                       temporary_block=temporary_block, shallow_fits_but_deep_does_not=fit_asymmetry,
                       shallow_fits_deep_extra_exceeds_budget=fit_asymmetry and d['extra_reserve_bytes'] > exp.budget))


def block_profile(exp, family):
    gap = [b - a for a, b in zip(exp.arrivals, exp.arrivals[1:])]
    depths = [max(choices) for choices in exp.eligible]
    return dict(family=family, requests=len(depths), with_checkpoint=sum(c > 0 for c in depths),
                multiple_checkpoints=sum(len(c) > 2 for c in exp.eligible), deepest=dict(Counter(depths)),
                planned_arrival_gap_ms=summarize(gap), input_tokens=summarize(exp.lengths))


def replay_study(mode, evidence, inputs, cost, cost_hash):
    groups, positions = defaultdict(Counter), defaultdict(set)
    margins = defaultdict(lambda: defaultdict(list))
    examples, blocks, exps, capacities = {}, {}, {}, defaultdict(set)
    nearest_comparisons = {s: Counter() for s in SSD_SCALES}
    serial_savings = {s: [] for s in SSD_SCALES}
    ssd_profiles = {s: scaled(cost, Fraction(s), Fraction(1)) for s in SSD_SCALES}
    variants = {}
    for run, schedule, family in archive_records(mode, evidence, inputs, cost_hash):
        block = run['block']
        key = block, run['capacity_bytes']
        capacities[block].add(key[1])
        if key not in exps:
            capacity = Fraction(key[1], 2**20)
            exps[key] = InstrumentedExperiment(dict(schedule['trace'], cost=cost), schedule['model'], capacity)
            variants[key] = {s: Experiment(dict(schedule['trace'], cost=profile), schedule['model'], capacity)
                             for s, profile in ssd_profiles.items()}
            if block not in blocks:
                blocks[block] = block_profile(exps[key], family)
        exp = exps[key]
        actual = list(map(Fraction, run['observed_arrivals_ms']))
        if sorted(r['snapshot']['request'] for r in run['decisions']) != list(range(len(exp.requests))):
            raise ValueError(f'incomplete decision receipts: {key}')
        for row in run['decisions']:
            obs = reconstruct(row['snapshot'])
            i = obs.request
            choices = exp.eligible[i]
            base = max(choices)
            if row['action'] != base:
                raise ValueError(f'archived policy did not choose deepest: {block}/{i}')
            exp.features = {}
            for group in ('all', family):
                positions[group].add((block, i))
            if base:
                expose(exp, obs, actual)
                if choose_observed(exp, 'local-cost', obs) != base or exp.inspect(obs, exact=True) != base:
                    raise ValueError(f'frozen replay did not choose deepest: {block}/{i}')
                if exp.forecasts:
                    ref = dict(block=block, request=i, policy=run['policy'], baseline=base,
                               case=run.get('service_case', run.get('round')))
                    compute = {c: exp.recompute(i, c) for c in choices}
                    for result in compare_window(exp, obs, base, choices, ref, compute, margins):
                        groups['all'].update(dict(explicit_pairs=1, positive_gap_relief=result['gap_relief_ms'] > 0,
                                                  wins_on_window=result['window_delta_ms'] < 0,
                                                  ties_on_window=result['window_delta_ms'] == 0,
                                                  some_earlier_gain=result['ahead_gain'], target_gain=result['target_gain'],
                                                  some_followers_gain=result['followers_gain']))
                        for field, comparator in (('relief_fraction', max), ('window_delta_ms', min)):
                            if field not in examples or comparator(result[field], examples[field][field]) != examples[field][field]:
                                examples[field] = result
                else:
                    groups['all'].update(isolated_snapshots=1, isolated_pairs=len(choices) - 1)
                for s, variant in variants[key].items():
                    expose(variant, obs, actual)
                    local = choose_observed(variant, 'local-cost', obs)
                    nearest_comparisons[s][f'{base}/{local}'] += 1
                    if local != base:
                        saved = sum(serial_components(variant, i, base)) - sum(serial_components(variant, i, local))
                        if saved <= 0:
                            raise ValueError('local-cost must improve over the deepest candidate to change its action')
                        serial_savings[s].append(saved)
            for group in ('all', family):
                note_snapshot(groups[group], exp, obs, choices, base)
    expected = {'primary': 9600, 'service': 3744}[mode]
    all_counts = groups['all']
    if (all_counts['snapshots'] != expected or sum(p['requests'] for p in blocks.values()) != len(positions['all'])
            or all_counts['wins_on_window'] or all_counts['ties_on_window']):
        raise ValueError(f'unexpected decision population or rankings: {mode}')
    nearest = {s: dict(deepest_local_action_pair_counts=dict(nearest_comparisons[s]),
                       disagreements=len(serial_savings[s]),
                       serial_saving_ms_on_disagreement=summarize(serial_savings[s]) if serial_savings[s] else None)
               for s in SSD_SCALES}
    return dict(snapshots={k: dict(v) for k, v in groups.items()}, distinct_positions={k: len(v) for k, v in positions.items()},
                blocks=blocks, capacity_bytes_by_block={k: sorted(v) for k, v in capacities.items()},
                pair_distributions={k: describe(v) for k, v in margins.items()},
                deepest_against_local_ssd_sensitivity=nearest,
                examples={k: {field: float(v) if isinstance(v, Fraction) else v for field, v in example.items()}
                          for k, example in examples.items()})


def forward_sweep(evidence, inputs, capacities, value_pairs, profiles):
    """Change one input at a time and let future states evolve under each policy."""
    lock = read(evidence / 'r10-local-holdout-execution-lock.json')
    results = {label: {} for label, _, _ in value_pairs}
    for block in lock['blocks']:
        block_id = block['id']
        raw = inputs.read(f'plans/{block_id}.schedule.json')
        if sha256(raw).hexdigest() != block['schedule_sha256']:
            raise ValueError(f'changed plan: {block_id}')
        plan = json.loads(raw)
        capacity_bytes, = capacities[block_id]
        capacity = Fraction(capacity_bytes, 2**20)
        for label, arrival_scale, prefill_scale in value_pairs:
            exp = Experiment(dict(plan['trace'], cost=profiles[prefill_scale]), plan['model'], capacity)
            start = exp.arrivals[0]
            exp.arrivals = [start + (t - start) * Fraction(arrival_scale) for t in exp.arrivals]
            actions, counts = {}, Counter()

            def observe_local(obs, action):
                i = obs.request
                actions[i] = action
                n = len(exp.eligible[i]) - 1
                counts.update(dict(requests=1, with_checkpoint=n > 0,
                                   local_below_deepest=action < max(exp.eligible[i]),
                                   with_checkpoint_ahead=n > 0 and i > obs.head,
                                   with_checkpoint_followers=n > 0 and obs.arrived > i + 1,
                                   with_checkpoint_inflight=n > 0 and obs.transfer_object is not None,
                                   multiple_and_inflight=n > 1 and obs.transfer_object is not None))

            def observe_entry(obs, action):
                counts['divergent_actions'] += action != actions[obs.request]
                counts['entry_decisions'] += 1
                counts['entry_below_deepest'] += action < max(exp.eligible[obs.request])

            audit_simulated_decisions(exp, 'local-cost', observe_local)
            audit_simulated_decisions(exp, 'entry-replay', observe_entry)
            if counts['requests'] != block['requests'] or counts['entry_decisions'] != block['requests']:
                raise ValueError(f'incomplete forward sweep {block_id}/{label}')
            results[label][block_id] = dict(counts)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=Path('docs/evidence'))
    parser.add_argument('--raw', type=Path, default=Path('data/raw/bailian'))
    parser.add_argument('--inputs', type=Path, default=Path('outputs/r10/local-reanalysis-inputs-v1.zip'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    input_hash = sha(args.inputs)
    if args.out.exists() or input_hash != read(args.evidence / 'r10-local-reanalysis-inputs-v1-summary.json')['archive_sha256']:
        raise ValueError('requires the frozen input archive and a new output path')
    with ZipFile(args.inputs) as archive:
        profile = archive.read('settings/r10-local-holdout-cost-v1.json')
        cost_hash = sha256(profile).hexdigest()
        cost = json.loads(profile)['cost']
        replay = {mode: replay_study(mode, args.evidence, archive, cost, cost_hash) for mode in ('primary', 'service')}
        capacities = replay['primary']['capacity_bytes_by_block']
        profiles = {'1': cost, **{s: scaled(cost, Fraction(1), Fraction(s)) for s in PREFILL_SCALES[1:]}}
        arrival = forward_sweep(args.evidence, archive, capacities, [(s, s, '1') for s in ARRIVAL_SCALES], profiles)
        prefill = forward_sweep(args.evidence, archive, capacities, [(s, '1', s) for s in PREFILL_SCALES[1:]], profiles)
        prefill['1'] = arrival['1']
    output = dict(input_archive_sha256=input_hash, cost_profile_sha256=cost_hash, replay=replay,
                  arrival_sweep=arrival, prefill_sweep=prefill,
                  sources=source_study(args.raw, args.evidence))
    with args.out.open('x', encoding='utf-8', newline='\n') as out:
        print(json.dumps(output, indent=2, sort_keys=True), file=out)


if __name__ == '__main__':
    main()
