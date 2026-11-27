"""Explain the delivered R10 service results without changing the frozen analysis."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

from scripts.gpu.analysis.explain_r10_primary_actions import explain
from scripts.gpu.analysis.summarize_r10_service_calibration import anchors
from scripts.gpu.experiments.r10_holdout_lock import POLICIES
from scripts.gpu.experiments.r10_service_lock import check_environment, load
from scripts.validation.check_r10_native_catalog import digest


def read(path):
    return json.loads(path.read_bytes())


def load_match(queue, rules):
    # Report every applicable calibration label, without reassigning the lock.
    found = anchors([{'queue_window': queue, 'rate': 'observed'}], rules)
    return [label for label, value in found.items() if value is not None]


def client_counts(arm, expected):
    client = arm['client']
    if arm['failed']:
        counts = Counter(client['statuses'])
        successes = client['slo_success_count']
        recorded_expected = client['expected']
    else:
        counts = Counter(terminal=client['completed'])
        successes = client['slo']['success_count']
        recorded_expected = client['completed']
    if recorded_expected != expected or sum(counts.values()) != expected or not 0 <= successes <= counts['terminal']:
        raise ValueError('client or SLO population differs from registered input')
    return dict(expected=expected, terminal=counts['terminal'], statuses=dict(counts), slo_success_count=successes,
                without_visible_output=None if arm['failed'] else client['requests_without_visible_output'])


def record_actions(record):
    ordered = sorted(record['decisions'], key=lambda r: r['snapshot']['request'])
    if [r['snapshot']['request'] for r in ordered] != list(range(len(ordered))):
        raise ValueError('wrong set of request identities in saved actions')
    return [row['action'] for row in ordered]


def collect(lock, official, delivery, archive, plans, extra_plans, cost, rules):
    blocks = {row['id']: row for row in lock['blocks']}
    schedules = {}
    for ident, b in blocks.items():
        path = (extra_plans if b['family'] == 'oasst' else plans) / (ident + '.schedule.json')
        if digest(path) != b['schedule_sha256']:
            raise ValueError(f'{ident}: changed registered schedule')
        schedules[ident] = read(path)
    groups = defaultdict(Counter)
    receipts = []
    for case, result in zip(lock['cases'], official['pairs'], strict=True):
        if any(case[k] != result[k] for k in case):
            raise ValueError('official summary is not in the frozen case order')
        ident, name = case['block'], case['id']
        b = blocks[ident]
        group = f"{b['family']}/{case['profile']}"
        ledger = json.loads(archive.read(f'{name}/pair.json'))
        if ledger['service_lock_sha256'] != official['lock_sha256'] or any(ledger[k] != v for k, v in case.items()):
            raise ValueError(f'{name}: archived pair does not match the execution lock')
        arms, choices = {}, {}
        for i, (policy, arm) in enumerate(zip(case['arm_order'], ledger['arms'], strict=True), 1):
            label = f'{i}-{policy}'
            if (arm['policy'], arm['prefix'], arm['order']) != (policy, label, i):
                raise ValueError(f'{name}/{label}: wrong arm identity')
            result_arm = result['arms'][policy]
            failed = bool(arm['return_code'])
            if result_arm['failed'] is not failed:
                raise ValueError(f'{name}/{label}: completion status disagrees with the summary')
            client = client_counts(result_arm, b['requests'])
            groups[group].update({f'{policy}/expected': client['expected'], f'{policy}/terminal': client['terminal'],
                                  f'{policy}/slo_success_count': client['slo_success_count'], f'{policy}/failed_arms': failed})
            row = dict(client=client, return_code=arm['return_code'])
            if not failed:
                name_in_zip = f'{name}/{label}.run.json'
                raw = archive.read(name_in_zip)
                if sha256(raw).hexdigest() != arm['files']['.run.json']:
                    raise ValueError(f'changed archived receipt: {name_in_zip}')
                record = json.loads(raw)
                if (record['block'], record['service_case'], record['policy'], record['service_lock_sha256'], record['seal_sha256']) != (
                        ident, name, policy, official['lock_sha256'], delivery['catalogues'][ident]['seal_sha256']):
                    raise ValueError(f'wrong archived arm identity: {name_in_zip}')
                action = explain(record, schedules[ident], cost)
                windows = load_match(result_arm['queue'], rules)
                row.update(observed_load_matches=windows, registered_anchor_observed=case['anchor'] in windows,
                           queue_window=result_arm['queue'], action=action)
                choices[policy] = record_actions(record)
                groups[group].update({f'{policy}/anchor_matched_arms': row['registered_anchor_observed'],
                                      **{f'{policy}/{key}': value for key, value in action['counts'].items()}})
            arms[policy] = row
        matched_actions = None
        if set(choices) == set(POLICIES):
            differences = [dict(request=i, local_cost=a, entry_replay=b) for i, (a, b) in
                           enumerate(zip(*(choices[p] for p in POLICIES), strict=True)) if a != b]
            matched_actions = dict(requests=b['requests'], differences=differences)
            groups[group].update(paired_action_requests=b['requests'], paired_action_disagreements=len(differences))
        comparable = bool(result['paired'] and result['paired']['valid_performance_comparison'])
        groups[group].update(registered_pairs=1, comparable_pairs=comparable)
        receipts.append(dict(case, family=b['family'], comparable_performance=comparable, paired_actions=matched_actions, arms=arms))
    return dict(by_family_profile={k: dict(v) for k, v in groups.items()}, cases=receipts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ('lock', 'summary', 'archive', 'delivery', 'plans', 'extra-plans', 'cost', 'calibration-design', 'out'):
        parser.add_argument('--' + k, type=Path, required=True)
    a = parser.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    lock = load(a.lock)
    check_environment(lock)
    official, delivery = read(a.summary), read(a.delivery)
    identity = (digest(a.lock), digest(a.archive), digest(a.summary))
    if (official['lock_sha256'], official['archive_sha256']) != identity[:2] or (
            delivery['lock_sha256'], delivery['archive_sha256'], delivery['summary_sha256']) != identity or (
            delivery['completed_pairs'], digest(a.cost), digest(a.calibration_design)) != (
                len(lock['cases']), lock['cost_profile_sha256'], lock['calibration_design_sha256']):
        raise ValueError('requires complete audited service evidence and frozen cost and calibration rules')
    with ZipFile(a.archive) as archive:
        result = collect(lock, official, delivery, archive, a.plans, a.extra_plans,
                         read(a.cost)['cost'], read(a.calibration_design)['anchor_rules'])
    with a.out.open('x', encoding='utf8', newline='\n') as file:
        print(json.dumps(dict(scope='descriptive load-label check and same-snapshot chooser replay, not a performance gate',
                              lock_sha256=identity[0], archive_sha256=identity[1], summary_sha256=identity[2],
                              delivery_sha256=digest(a.delivery), **result), sort_keys=True, indent=2), file=file)


if __name__ == '__main__':
    main()
