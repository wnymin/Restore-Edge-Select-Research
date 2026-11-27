"""Audit finished service pairs and report client latency and SLO goodput."""

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path
from statistics import mean
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from scripts.gpu.analysis.audit_r10_service_pipe import audit, audit_failed_sse, distribution, first_visible_clock
from scripts.gpu.analysis.summarize_r10_pilot_load import measured_window
from scripts.gpu.experiments.r10_service_lock import analysis_hashes, load
from scripts.validation.check_r10_native_catalog import digest


def read(p):
    return json.loads(p.read_bytes())


def selected_paths(receipt, schedule, dispatch):
    trace = schedule['trace']
    eligible = [set(row['boundary_tokens'] for row in request['candidates']) for request in trace['requests']]
    decisions = {x['snapshot']['request']: x['action'] for x in receipt['decisions']}
    assert set(decisions) == set(range(len(eligible))) and all(action in eligible[i] for i, action in decisions.items())
    recompute = [i for i, action in decisions.items() if action == 0]
    objects = {str(row['id']): row for row in trace['objects']}
    uploads = Counter()
    for t in receipt['transfers']:
        read_info = t['read']
        obj = objects[str(read_info['object_id'])]
        uploads[obj['tier'] + '/' + read_info['lane_use']] += obj['bytes']
    assert dispatch['peak_prefetch_bytes'] == receipt['peak_prefetch_bytes']
    return dict(selected_boundary_counts=dict(Counter(decisions.values())),
                recompute_no_nonzero_candidate=sum(not any(eligible[i]) for i in recompute),
                selected_recompute_with_nonzero_candidate=sum(any(eligible[i]) for i in recompute),
                peak_stage_bytes=dispatch['peak_prefetch_bytes'],
                logical_upload_bytes_by_tier_and_trigger=dict(uploads),
                device_event_counts=dispatch['device_event_counts'])


def curve_cell(rows):
    comparable = [r for r in rows if r['paired'] and r['paired']['valid_performance_comparison']]
    xs = [r['paired']['goodput_gain_per_s'] for r in comparable]
    def point(row, policy):
        arm = row['arms'][policy]
        client = arm['client']
        return dict(block=row['paired']['block'], offered_per_s=arm['queue']['offered_arrivals_per_s'],
                    p95_visible_ttft_ms=client['ttft_visible_ms'].get('p95_ms'),
                    p95_frame_tpot_ms=client['token_frame_tpot_ms'].get('p95_ms'),
                    completed_per_s=client['completions_per_s'], slo_success_fraction=client['slo']['success_fraction'],
                    goodput_per_s=client['slo']['goodput_per_s'])
    return dict(registered_pairs=len(rows), comparable_pairs=len(xs),
                failed_pairs=sum(any(a['failed'] for a in r['arms'].values()) for r in rows),
                different_output_pairs=sum(bool(r['paired'] and r['paired']['different_greedy_outputs']) for r in rows),
                mean_goodput_gain_per_s=mean(xs) if xs else None, observed_range=[min(xs), max(xs)] if xs else None,
                points=[{k: r['paired'][k] for k in ('block', 'offered_per_s', 'p95_visible_ttft_change_ms',
                        'p95_frame_tpot_change_ms', 'slo_success_fraction_change', 'goodput_gain_per_s')}
                        for r in comparable],
                policy_points={policy: [point(r, policy) for r in comparable] for policy in ('local-cost', 'entry-replay')})


def paired_service(baseline, entry, block):
    if baseline['failed'] or entry['failed']:
        return None
    lc, ec = baseline['client'], entry['client']
    difference = [b - a for a, b in zip(baseline['request_ttft'], entry['request_ttft'], strict=True) if a is not None and b is not None]
    mismatched = sum(a != b for a, b in zip(baseline['token_output'], entry['token_output'], strict=True))
    def delta(metric):
        a, b = (client[metric].get('p95_ms') for client in (lc, ec))
        return b - a if a is not None and b is not None else None
    return dict(block=block, paired_visible_ttft_ms=distribution(difference),
                max_individual_visible_ttft_regression_ms=max(difference) if difference else None,
                different_greedy_outputs=mismatched, valid_performance_comparison=mismatched == 0,
                offered_per_s={p: arm['queue']['offered_arrivals_per_s'] for p, arm in
                               (('local-cost', baseline), ('entry-replay', entry))},
                p95_visible_ttft_change_ms=delta('ttft_visible_ms'), p95_frame_tpot_change_ms=delta('token_frame_tpot_ms'),
                p95_frame_itl_change_ms=delta('inter_token_frame_ms'),
                slo_success_fraction_change=ec['slo']['success_fraction'] - lc['slo']['success_fraction'],
                goodput_gain_per_s=ec['slo']['goodput_per_s'] - lc['slo']['goodput_per_s'])


def evaluate(prefix, schedule, ledger, block, case, lock, lock_hash):
    slo = lock['experimental_slo']
    target = slo['visible_ttft_ms'], slo['client_request_tpot_ms']
    if ledger['return_code']:
        path = prefix.with_suffix('.failure.json')
        receipt = read(path) if path.exists() else {}
        reason = {k: receipt[k] for k in ('error_type', 'error', 'worker_return_code', 'terminated_by_controller') if k in receipt}
        return dict(failed=True, return_code=ledger['return_code'], failure=reason,
                    client=audit_failed_sse(receipt, block['requests'], *target), token_output=None, request_ttft=None)
    receipt = read(prefix.with_suffix('.run.json'))
    assert receipt['scope'] == 'R10 local streaming service holdout arm' and not receipt['qualification']
    assert (receipt['block'], receipt['service_case'], receipt['service_lock_sha256'], receipt['policy']) == (
        block['id'], case['id'], lock_hash, ledger['policy'])
    assert receipt['observer_sha256'] == lock['environment']['observer_sha256'] and receipt['worker_sha256'] == lock['environment']['worker_sha256']
    assert receipt['cost_profile_sha256'] == lock['cost_profile_sha256'] and receipt['schedule_sha256'] == digest(schedule)
    assert receipt['capacity_bytes'] == Fraction(lock['capacity_mib']) * 2**20 and receipt['warmup_grid']
    assert (receipt['service']['short_limit'], receipt['service']['long_limit']) == (case['short_tokens'], case['long_tokens'])
    schedule_data = read(schedule)
    arrivals = [Fraction(str(r['arrival_ms'])) for r in schedule_data['trace']['requests']]
    expected = [(x - arrivals[0]) * Fraction(case['arrival_scale']) for x in arrivals]
    assert list(map(Fraction, receipt['planned_arrival_offsets_ms'])) == expected
    offered, completed = (list(map(Fraction, receipt[k])) for k in ('client_arrivals_ms', 'completions_ms'))
    audited = audit(prefix, schedule, *target)
    assert audited['total_requests'] == block['requests']
    token_output, request_ttft = [], []
    for i, r in enumerate(receipt['service']['http_sse_requests']):
        row = r['events']
        token_output.append(dict(reason=row[-1]['reason'], tokens=[e['id'] for e in row[:-1]]))
        first = first_visible_clock(row)
        request_ttft.append(float(first - offered[i]) if first is not None else None)
    observed = list(map(Fraction, receipt['observed_arrivals_ms']))
    starts = [Fraction(row['observed_ms']) for row in receipt['starts']]
    assert len(offered) == len(observed) == len(starts) == len(completed) == block['requests']
    assert all(a <= o <= s <= c for a, o, s, c in zip(offered, observed, starts, completed, strict=True))
    pipe_fields = ('native_prefill_done_to_first_emission', 'end_reason', 'output_lengths')
    return dict(failed=False, client=audited['http_sse'], queue=measured_window(offered, completed),
                observed_arrival_to_compute_issued_ms=distribution([float(s - o) for s, o in zip(starts, observed, strict=True)]),
                worker={k: audited[k] for k in pipe_fields},
                recovery=selected_paths(receipt, schedule_data, audited['dispatch']),
                token_output=token_output, request_ttft=request_ttft)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('lock', 'plans', 'extra-plans', 'campaign', 'archive', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.archive.exists() or a.out.exists():
        raise FileExistsError('select new service evidence paths')
    lock, lock_hash = load(a.lock), digest(a.lock)
    if lock['environment']['service_analysis_sha256'] != analysis_hashes():
        raise ValueError('service analysis differs from the frozen analysis')
    blocks = {r['id']: r for r in lock['blocks']}
    marker = a.campaign / 'campaign.json'
    assert read(marker)['service_lock_sha256'] == lock_hash
    raw, results, groups = {'campaign.json': marker, 'execution-lock.json': a.lock}, [], defaultdict(list)
    for log in sorted(a.campaign.glob('*.prepare.log')):
        raw[log.name] = log
    for case in lock['cases']:
        b = blocks[case['block']]
        files = a.extra_plans if b['family'] == 'oasst' else a.plans
        schedule = files / (b['id'] + '.schedule.json')
        assert digest(schedule) == b['schedule_sha256']
        folder = a.campaign / case['id']
        manifest = folder / 'pair.json'
        ledger = read(manifest)
        raw[f'{folder.name}/pair.json'] = manifest
        assert ledger['service_lock_sha256'] == lock_hash and all(ledger[k] == v for k, v in case.items())
        arms = {}
        for i, (policy, arm) in enumerate(zip(case['arm_order'], ledger['arms'], strict=True), 1):
            assert (arm['order'], arm['policy'], arm['prefix']) == (i, policy, f'{i}-{policy}')
            prefix = folder / arm['prefix']
            receipt = prefix.with_suffix('.arm.json')
            assert read(receipt) == arm
            for key, h in arm['files'].items():
                path = prefix.with_suffix(key)
                assert digest(path) == h
                raw[f'{folder.name}/{path.name}'] = path
            raw[f'{folder.name}/{receipt.name}'] = receipt
            arms[policy] = evaluate(prefix, schedule, arm, b, case, lock, lock_hash)
        compare = paired_service(arms['local-cost'], arms['entry-replay'], b['id'])
        for arm in arms.values():
            arm.pop('token_output')
            arm.pop('request_ttft')
        entry = dict(case, family=b['family'], arms=arms, paired=compare)
        results.append(entry)
        groups[b['family'], case['profile'], case['anchor']].append(entry)
    curves = {'/'.join(key): curve_cell(rows) for key, rows in groups.items()}
    with ZipFile(a.archive, 'x') as z:
        for label, path in raw.items():
            z.writestr(ZipInfo(label), path.read_bytes(), compress_type=ZIP_DEFLATED)
    with a.out.open('x', encoding='utf8') as f:
        print(json.dumps(dict(scope='finite open-loop HTTP/SSE runs on one slot; descriptive paired blocks, not a sustained capacity estimate',
                              lock_sha256=lock_hash, archive_sha256=digest(a.archive), curves=curves, pairs=results),
                         sort_keys=True, indent=2), file=f)


if __name__ == '__main__':
    main()
