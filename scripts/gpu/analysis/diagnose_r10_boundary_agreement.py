"""Replay the R10 candidate rankings from the frozen holdout archives."""

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from zipfile import ZipFile

from restore_edge_select.live_snapshot import choose_observed
from restore_edge_select.simulator import Experiment
from scripts.gpu.analysis.explain_r10_controlled_choices import scaled
from scripts.gpu.analysis.replay_r10_calibration_decisions import reconstruct
from scripts.gpu.analysis.summarize_r10_holdout import percentile
from scripts.gpu.experiments.r10_holdout_lock import check_controller
from scripts.validation.check_r10_native_catalog import digest


SCENARIOS = (('13/10', '1'), ('2', '1'), ('4', '1'), ('8', '1'), ('16', '1'),
             ('1', '3/4'), ('1', '1/2'), ('1', '1/4'), ('8', '1/4'))


class InstrumentedExperiment(Experiment):
    def inspect(self, obs, exact=False):
        self.exact = exact
        self.scored = False
        self.forecasts = {}
        self.features = {}
        self.request = obs.request
        return choose_observed(self, 'entry-replay', obs)

    def _run(self, policy, threshold, boundaries, max_combinations, n, **opts):
        if opts.get('decision_cursor') is not None and policy == 'snapshot-screened':
            if self.exact:
                # Bypass the screen and lower-bound pruning, keeping R10's
                # fixed followers and the same lean event replay.
                opts.update(screen_mode='classified', screen_search='lean', screen_selector=self.capture)
            else:
                score = opts['score_candidate']

                def counted(row):
                    self.scored = True
                    return score(row)

                opts['score_candidate'] = counted
        result = super()._run(policy, threshold, boundaries, max_combinations, n, **opts)
        if policy == 'fixed':
            self.forecasts[boundaries[self.request]] = result['completion_ms']
        return result

    def capture(self, features, fallback, visible_followers):
        self.features = features
        return 4


def serial_components(exp, request, boundary):
    deps = exp.candidates[request][boundary]
    compute = exp.recompute(request, boundary)
    copy = (exp.path_time('device_copy', exp.objects[deps[-1]]['bytes'])
            if deps and deps[-1] in exp.initial and boundary < exp.lengths[request] else 0)
    return compute, copy + sum(exp.transfer_time(obj) for obj in deps if obj not in exp.initial)


def expose(exp, obs, actual):
    exp.arrivals = actual[:obs.arrived] + [obs.now_ms + 1] * (len(actual) - obs.arrived)


def summarize(values):
    return {k: round(float(v), 6) for k, v in dict(min=min(values), p10=percentile(values, .1),
            median=median(values), p90=percentile(values, .9), max=max(values)).items()}


def archive_records(mode, evidence, inputs, cost_hash):
    phase = 'service-' if mode == 'service' else ''
    lock_raw = (evidence / f'r10-local-{phase}holdout-execution-lock.json').read_bytes()
    lock = json.loads(lock_raw)
    lock_hash = sha256(lock_raw).hexdigest()
    check_controller(lock)
    if lock['cost_profile_sha256'] != cost_hash:
        raise ValueError(f'{mode}: changed cost profile')
    blocks = {b['id']: b for b in lock['blocks']}
    stem = evidence / f'r10-native-{mode}-holdout'
    summary = json.loads(stem.with_name(stem.name + '-summary.json').read_bytes())
    archive = stem.with_suffix('.zip')
    if (summary['lock_sha256'], summary['archive_sha256']) != (lock_hash, digest(archive)):
        raise ValueError(f'{mode}: changed archive or lock')
    schedules = {}
    for block, metadata in blocks.items():
        folder = 'service-plans' if phase and metadata['family'] == 'oasst' else 'plans'
        raw = inputs.read(f'{folder}/{block}.schedule.json')
        if sha256(raw).hexdigest() != metadata['schedule_sha256']:
            raise ValueError(f'changed schedule: {block}')
        schedules[block] = json.loads(raw)
    with ZipFile(archive) as zipped:
        if zipped.read('execution-lock.json') != lock_raw:
            raise ValueError(f'{mode}: archive contains another lock')
        for member in sorted(zipped.namelist()):
            if member.endswith('.run.json'):
                run = json.loads(zipped.read(member))
                block = run['block']
                lock_key = 'service_lock_sha256' if phase else 'holdout_lock_sha256'
                if (run[lock_key], run['schedule_sha256'], run['cost_profile_sha256']) != (
                        lock_hash, blocks[block]['schedule_sha256'], cost_hash):
                    raise ValueError(f'changed execution identity: {member}')
                yield run, schedules[block], blocks[block]['family']


def window_deltas(exp, obs, local):
    forecasts = exp.forecasts
    if forecasts.keys() != set(exp.eligible[obs.request]) or any(
            len(ends) != min(obs.arrived, obs.request + 5) - obs.head for ends in forecasts.values()):
        raise ValueError('expected every candidate in the same visible window')
    target = obs.request - obs.head
    basis = forecasts[local]
    others = set(forecasts) - {local}
    other = min(others, key=lambda c: (sum(forecasts[c]), -c))
    delta = [a - b for a, b in zip(forecasts[other], basis, strict=True)]
    follower_gain = any(sum(forecasts[c][target+1:]) < sum(basis[target+1:]) for c in others)
    target_gain = any(forecasts[c][target] < basis[target] for c in others)
    return other, delta[target], sum(delta[:target]), sum(delta[target+1:]), follower_gain, target_gain


def analyze(mode, evidence, inputs, cost, cost_hash):
    counts, depths = Counter(), Counter()
    exposures, families, gaps = defaultdict(Counter), defaultdict(Counter), defaultdict(list)
    sensitivity = {pair: Counter() for pair in SCENARIOS}
    migrations = defaultdict(Counter)
    examples = {}
    unique = defaultdict(set)
    nearest = None
    rates = {pair: scaled(cost, *map(Fraction, pair)) for pair in SCENARIOS}
    experiments = {}
    for run, schedule, family in archive_records(mode, evidence, inputs, cost_hash):
        key = (run['block'], run['capacity_bytes'])
        if key not in experiments:
            capacity = Fraction(run['capacity_bytes'], 2**20)
            trace, model = schedule['trace'], schedule['model']
            experiments[key] = (InstrumentedExperiment(dict(trace, cost=cost), model, capacity),
                                {pair: Experiment(dict(trace, cost=c), model, capacity) for pair, c in rates.items()})
        exp, variants = experiments[key]
        actual = list(map(Fraction, run['observed_arrivals_ms']))
        if sorted(r['snapshot']['request'] for r in run['decisions']) != list(range(len(exp.requests))):
            raise ValueError(f'incomplete receipt for {key}')
        counts['arms'] += 1
        for row in run['decisions']:
            obs = reconstruct(row['snapshot'])
            i = obs.request
            choices = exp.eligible[i]
            counts['snapshots'] += 1
            nonzero = len(choices) - 1
            families[family].update(['snapshots', f'{min(nonzero, 2)}_nonzero_group'])
            if not nonzero:
                if row['action'] != 0:
                    raise ValueError('recorded an impossible boundary without a checkpoint')
                counts['only_recompute'] += 1
                continue
            counts['with_checkpoint'] += 1
            counts['multiple_checkpoints'] += nonzero > 1
            expose(exp, obs, actual)
            local = choose_observed(exp, 'local-cost', obs)
            entry = exp.inspect(obs)
            if (local if run['policy'] == 'local-cost' else entry) != row['action']:
                raise ValueError(f'online action did not replay: {key}/{i}')
            counts['frozen_disagreements'] += entry != local
            counts['local_chooses_deepest'] += local == max(choices)
            route = 'replay' if exp.forecasts else ('screen' if exp.scored else 'isolated')
            counts[route] += 1
            full = exp.inspect(obs, exact=True)
            counts['full_differs_from_entry'] += full != entry
            counts[f'{route}_full_differs_from_entry'] += full != entry

            serial = {c: serial_components(exp, i, c) for c in choices}
            ranked = sorted(choices, key=lambda c: (sum(serial[c]), -c))
            if local != ranked[0]:
                raise ValueError('serial decomposition does not reproduce local-cost')
            runner = ranked[1]
            compute_saved = serial[runner][0] - serial[local][0]
            extra_io = serial[local][1] - serial[runner][1]
            depths[f'{local}/{runner}'] += 1
            if runner:
                ssd_costs = [sum(exp.transfer_time(obj) for obj in exp.candidates[i][c]
                                 if obj not in exp.initial and exp.objects[obj]['tier'] == 'SSD')
                             for c in (local, runner)]
                counts['nonzero_runner_up_same_serial_ssd_cost'] += ssd_costs[0] == ssd_costs[1]
            gaps['serial_compute_saved'].append(compute_saved)
            gaps['serial_extra_io_and_copy'].append(extra_io)
            margin = compute_saved - extra_io
            gaps['serial_margin'].append(margin)
            counts['serial_margin_at_most_10ms'] += margin <= 10
            features = exp.features.get(local, {})
            flags = dict(visible_follower=obs.arrived > i + 1, ahead_of_head=i > obs.head,
                         active_compute=obs.compute_start_ms is not None, transfer_inflight=obs.transfer_object is not None,
                         active_with_zero_predicted_compute=(obs.compute_start_ms is not None and features['head_slack_ms'] == 0),
                         winner_reserved_overlap=bool(features.get('shared_reservation_bytes')),
                         candidate_temporary_capacity_block=any(c and i > obs.head and not d['prefetch_fits']
                                                                     and d['extra_reserve_bytes'] <= exp.budget
                                                                     for c, d in exp.features.items()))
            for name, flag in flags.items():
                if flag:
                    exposures[name].update(snapshots=1, multiple_checkpoints=nonzero > 1)
            if exp.forecasts:
                other, target, ahead, follower, follower_gain, target_gain = window_deltas(exp, obs, local)
                total = target + ahead + follower
                gaps['full_target_delta'].append(target)
                gaps['full_window_delta'].append(total)
                counts['some_alternative_improves_follower_sum'] += follower_gain
                counts['some_alternative_improves_target'] += target_gain
                if nearest is None or total < nearest[0]:
                    nearest = (total, dict(block=run['block'], case=run.get('service_case', run.get('round')),
                                           policy=run['policy'], request=i, local=local, alternative=other,
                                           target_delta_ms=float(target), ahead_delta_ms=float(ahead),
                                           follower_delta_ms=float(follower), window_delta_ms=float(total),
                                           visible_followers=min(obs.arrived - i - 1, 4),
                                           serial_margin_ms=float(sum(serial[other]) - sum(serial[local]))))
            for pair, variant in variants.items():
                expose(variant, obs, actual)
                a = choose_observed(variant, 'local-cost', obs)
                b = choose_observed(variant, 'entry-replay', obs)
                sensitivity[pair][f'{a}/{b}' if a != b else 'agree'] += 1
                if a != b:
                    unique[pair].add((run['block'], i))
                    fit_a = a > 0 and bool(exp.features.get(a, {}).get('prefetch_fits'))
                    fit_b = b > 0 and bool(exp.features.get(b, {}).get('prefetch_fits'))
                    migrations[pair][f'{a}/{b}:prefetch_fits={int(fit_a)}/{int(fit_b)}'] += 1
                    if pair not in examples:
                        examples[pair] = dict(block=run['block'], request=i, policy=run['policy'],
                                              case=run.get('service_case', run.get('round')),
                                              choices=[a, b], visible_followers=min(obs.arrived - i - 1, 4))

    if counts['snapshots'] != {'primary': 9600, 'service': 3744}[mode] or counts['frozen_disagreements']:
        raise ValueError(f'unexpected frozen replay population or actions: {mode}')
    cells = []
    for ssd, prefill in SCENARIOS:
        key = (ssd, prefill)
        cells.append(dict(ssd_multiplier=ssd, prefill_multiplier=prefill, choice_pair_counts=dict(sensitivity[key]),
                          distinct_block_request_positions_with_disagreement=len(unique[key]),
                          disagreements_by_prefetch_fit=dict(migrations[key]), first_disagreement=examples.get(key)))
    return dict(counts=dict(counts), by_family=dict(families), exposures=dict(exposures),
                local_and_serial_runner_up=dict(depths), margins_ms={k: summarize(v) for k, v in gaps.items()},
                nearest_full_window=nearest[1], sensitivity=cells)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=Path('docs/evidence'))
    parser.add_argument('--inputs', type=Path, default=Path('outputs/r10/local-reanalysis-inputs-v1.zip'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.evidence / 'r10-local-reanalysis-inputs-v1-summary.json').read_bytes())
    input_hash = digest(args.inputs)
    if manifest['archive_sha256'] != input_hash or args.out.exists():
        raise ValueError('requires the frozen input archive and a new output path')
    with ZipFile(args.inputs) as inputs:
        raw = inputs.read('settings/r10-local-holdout-cost-v1.json')
        cost_hash = sha256(raw).hexdigest()
        cost = json.loads(raw)['cost']
        results = {mode: analyze(mode, args.evidence, inputs, cost, cost_hash) for mode in ('primary', 'service')}
    output = dict(scope='fixed recorded snapshots, no GPU or latency prediction at changed costs',
                  input_archive_sha256=input_hash, cost_profile_sha256=cost_hash, modes=results)
    with args.out.open('x', encoding='utf-8', newline='\n') as out:
        print(json.dumps(output, indent=2, sort_keys=True), file=out)


if __name__ == '__main__':
    main()
