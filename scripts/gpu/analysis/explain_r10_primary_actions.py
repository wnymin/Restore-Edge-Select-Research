"""Replay both choosers after the complete, frozen R10 primary archive exists."""

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
from zipfile import ZipFile

from restore_edge_select.live_snapshot import choose_observed
from restore_edge_select.simulator import Experiment
from scripts.gpu.analysis.replay_r10_calibration_decisions import reconstruct
from scripts.gpu.analysis.summarize_r10_holdout import percentile
from scripts.gpu.experiments.r10_holdout_lock import POLICIES, check_controller, load
from scripts.validation.check_r10_native_catalog import digest


def explain(recorded, schedule, cost):
    trace = dict(schedule['trace'], cost=cost)
    exp = Experiment(trace, schedule['model'], Fraction(recorded['capacity_bytes'], 2**20))
    actual = list(map(Fraction, recorded['observed_arrivals_ms']))
    decisions = sorted(recorded['decisions'], key=lambda r: r['snapshot']['request'])
    if [r['snapshot']['request'] for r in decisions] != list(range(len(exp.requests))) or len(actual) != len(decisions):
        raise ValueError('missing or duplicate request snapshots')
    counts, action_counts, moves = Counter(), Counter(), Counter()
    disagreements = []
    for row in decisions:
        obs = reconstruct(row['snapshot'])
        if obs.request >= obs.arrived or obs.arrived > len(actual):
            raise ValueError('the recorded snapshot chooses an unavailable request')
        # Unarrived requests do not expose exact arrival times to either chooser.
        exp.arrivals = actual[:obs.arrived] + [obs.now_ms + 1] * (len(actual) - obs.arrived)
        local, entry = [choose_observed(exp, p, obs) for p in POLICIES]
        if row['action'] != (local if recorded['policy'] == 'local-cost' else entry):
            raise ValueError(f"online choice did not replay: {recorded['block']}/{obs.request}")
        choices = exp.eligible[obs.request]
        n_nonzero = sum(b > 0 for b in choices)
        counts['snapshots'] += 1
        counts['any_nonzero_candidate'] += n_nonzero > 0
        counts['multiple_nonzero_candidates'] += n_nonzero > 1
        counts['chose_zero_with_nonzero_candidate'] += row['action'] == 0 and n_nonzero > 0
        counts['same_snapshot_disagreements'] += local != entry
        action_counts[row['action']] += 1
        if local != entry:
            disagreements.append(dict(request=obs.request, local_cost=local, entry_replay=entry,
                                      eligible=list(choices), visible_queue=obs.arrived - obs.head,
                                      active_compute=obs.compute_start_ms is not None,
                                      active_transfer=obs.transfer_object is not None))
    objects = {str(r['id']): r for r in trace['objects']}
    for row in recorded['transfers']:
        read = row['read']
        obj = objects[str(read['object_id'])]
        moves[obj['tier'] + '/' + read['lane_use']] += obj['bytes']
    choose = [r['choose_ns'] / 1e6 for r in decisions]
    return dict(counts=dict(counts), chosen_boundary_counts=dict(action_counts),
                logical_uploaded_bytes_by_tier_and_trigger=dict(moves), same_snapshot_disagreements=disagreements,
                chooser_called_after_snapshot_ms=dict(mean=mean(choose), p50=percentile(choose, .5), p95=percentile(choose, .95)))


def preparation_seals(lock, lock_hash, summary, archive_path):
    if (summary['lock_sha256'], summary['archive_sha256']) != (lock_hash, digest(archive_path)):
        raise ValueError('preparation summary or archive belongs to a different primary freeze')
    seals = {row['block']: row['seal_sha256'] for row in summary['blocks']}
    if len(summary['blocks']) != len(seals) or set(seals) != {b['id'] for b in lock['blocks']}:
        raise ValueError('preparation archive must cover every locked block once')
    with ZipFile(archive_path) as z:
        if sha256(z.read('execution-lock.json')).hexdigest() != lock_hash:
            raise ValueError('preparation archive contains a different primary lock')
        for block, seal_hash in seals.items():
            if sha256(z.read(block + '.seal.json')).hexdigest() != seal_hash:
                raise ValueError(f'archived preparation seal changed: {block}')
    return seals


def collect(lock, plans, cost, official, archive, seals):
    expected = [(r['block'], r['round'], r['arm_order']) for r in lock['run_pairs']]
    registered = [(r['block'], r['round'], r['arm_order']) for r in official['pair_rows']]
    if registered != expected:
        raise ValueError('requires the complete official 60-pair result in the frozen order')
    blocks, grouped = {r['id']: r for r in lock['blocks']}, defaultdict(list)
    schedules = {}
    for ident, block in blocks.items():
        data = (plans / (ident + '.schedule.json')).read_bytes()
        if sha256(data).hexdigest() != block['schedule_sha256']:
            raise ValueError(f'schedule changed: {ident}')
        schedules[ident] = json.loads(data)
    arms = []
    for ident, rnd, order in expected:
        pair = f'r{rnd:02d}-{ident}'
        ledger = json.loads(archive.read(pair + '/pair.json'))
        if (ledger['block'], ledger['round'], ledger['arm_order'], ledger['lock_sha256']) != (ident, rnd, order, official['lock_sha256']):
            raise ValueError(f'archive pair does not match the primary lock: {pair}')
        for policy, arm, idx in zip(order, ledger['arms'], (1, 2), strict=True):
            if (arm['policy'], arm['order'], arm['prefix']) != (policy, idx, f'{idx}-{policy}'):
                raise ValueError(f'wrong arm identity: {pair}/{policy}')
            row = dict(block=ident, round=rnd, policy=policy, return_code=arm['return_code'])
            if not arm['return_code']:
                raw = archive.read(f"{pair}/{arm['prefix']}.run.json")
                if sha256(raw).hexdigest() != arm['receipt_sha256']:
                    raise ValueError(f'archived receipt changed: {pair}/{policy}')
                recorded = json.loads(raw)
                if (recorded['block'], recorded['round'], recorded['policy'], recorded['holdout_lock_sha256']) != (ident, rnd, policy, official['lock_sha256']):
                    raise ValueError(f'wrong receipt identity: {pair}/{policy}')
                if recorded['seal_sha256'] != seals[ident]:
                    raise ValueError(f'online run used a different preparation seal: {pair}/{policy}')
                row.update(explain(recorded, schedules[ident], cost))
            arms.append(row)
            grouped[ident, policy].append(row)
    by_block = []
    for (ident, policy), rows in grouped.items():
        counts, moves = Counter(), Counter()
        for row in rows:
            counts.update(row.get('counts', {}))
            moves.update(row.get('logical_uploaded_bytes_by_tier_and_trigger', {}))
        by_block.append(dict(block=ident, family=blocks[ident]['family'], source=blocks[ident]['source'], policy=policy,
                             successful_arms=sum(row['return_code'] == 0 for row in rows), failed_arms=sum(row['return_code'] != 0 for row in rows),
                             counts=dict(counts), logical_uploaded_bytes_by_tier_and_trigger=dict(moves)))
    return dict(block_policy=by_block, arms=arms)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('lock', 'plans', 'cost', 'primary-summary', 'archive', 'preparation-summary', 'preparation-archive', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    lock = load(a.lock)
    check_controller(lock)
    official = json.loads(a.primary_summary.read_bytes())
    lock_hash, archive_hash = digest(a.lock), digest(a.archive)
    if (official['lock_sha256'], official['archive_sha256'], lock['cost_profile_sha256']) != (lock_hash, archive_hash, digest(a.cost)):
        raise ValueError('official summary, archive, lock or cost provenance does not match')
    prep = json.loads(a.preparation_summary.read_bytes())
    seals = preparation_seals(lock, lock_hash, prep, a.preparation_archive)
    with ZipFile(a.archive) as z:
        if z.read('execution-lock.json') != a.lock.read_bytes():
            raise ValueError('archive contains a different primary lock')
        result = collect(lock, a.plans, json.loads(a.cost.read_bytes())['cost'], official, z, seals)
    with a.out.open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(dict(scope='descriptive same-snapshot replay, not counterfactual latency or an acceptance gate',
                              lock_sha256=lock_hash, archive_sha256=archive_hash, primary_summary_sha256=digest(a.primary_summary),
                              preparation_summary_sha256=digest(a.preparation_summary), preparation_archive_sha256=prep['archive_sha256'],
                              **result), sort_keys=True, indent=2), file=f)


if __name__ == '__main__':
    main()
