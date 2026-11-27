"""Index R10 local acceptance evidence, independently of positive policy results."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
from zipfile import ZipFile

from scripts.gpu.experiments import r10_holdout_lock as primary, r10_service_lock as service
from scripts.validation.check_r10_native_catalog import digest


def read(path):
    return json.loads(path.read_bytes())


def need(condition, message):
    if not condition:
        raise ValueError(message)


def member_hash(archive, name):
    with archive.open(name) as member:
        return file_digest(member, 'sha256').hexdigest()


def sealed(e, stem, files):
    summary, archive = (e / (stem + ext) for ext in ('-summary.json', '.zip'))
    data = read(summary)
    need(data['archive_sha256'] == digest(archive), f'{archive.name}: summary archive identity mismatch')
    with ZipFile(archive) as z:
        need(len(set(z.namelist())) == len(z.namelist()) and z.testzip() is None, f'{archive.name}: invalid archive members')
    files.update({p.name: digest(p) for p in (summary, archive)})
    return data


def available(e, names):
    return [name for name in names if not (e / name).is_file()]


def audit(e, include_staged_followups=False):
    files, phases, pending = {}, {}, {}

    def extra(name):
        p = e / name
        files[name] = digest(p)
        return read(p)

    p_lock = e / 'r10-local-holdout-execution-lock.json'
    s_lock = e / 'r10-local-service-holdout-execution-lock.json'
    p, s = primary.load(p_lock), service.load(s_lock)
    files.update({x.name: digest(x) for x in (p_lock, s_lock)})

    stage = extra('r10-stage-pool-summary.json')
    pool_zip = e / 'r10-stage-pool.zip'
    with ZipFile(pool_zip) as pool:
        need(member_hash(pool, 'stage-pool.csv') == stage['raw_sha256'] and json.loads(pool.read('summary.json')) == stage,
             'native stage pool summary and raw event record differ')
    files[pool_zip.name] = digest(pool_zip)
    mixed = sealed(e, 'r10-native-mixed-qualification', files)
    need(stage['formal_exact_logit_positions'] == 480 and set(stage['arms']) == {'exact-union', 'inflight-cancel', 'union-minus-page'} and
         len(mixed['fixtures']) == 3 and mixed['final_logits_matches'] == mixed['full_state_matches'] == 8,
         'incomplete R10-GPU-01 qualification evidence')
    phases['R10-GPU-01'] = dict(tested_exact_logit_positions=480, online_mixed_fixtures=3)

    grid, transfer, joint, budgets = (sealed(e, stem, files) for stem in (
        'r10-prefill-grid', 'r10-transfer-grid', 'r10-native-joint-grid', 'r10-native-active-capacity-sweep'))
    need(grid['samples'] == 432 and transfer['records'] == 48 and len(joint['cases']) == 8 and
         {x['capacity_bytes'] for x in budgets['arms']} == {67108864, 86245376, 150994944} and len(p['calibration_aa']) == 5,
         'incomplete R10-GPU-02 calibration coverage')
    phases['R10-GPU-02'] = dict(prefill_samples=432, transfer_samples=48, joint_cases=8, capacity_tiers=3, aa_sources=5)

    mechanisms = sealed(e, 'r10-native-mechanism-replications', files)['pairs']
    native, injected = (mechanisms[name] for name in ('repeat-shared-tight', 'repeat-shared-soft700'))
    key = 'shallow_minus_deep_total_response_ms'
    need(len(mechanisms) == 7 and native['schedule_sha256'] == injected['schedule_sha256'] and
         native['soft_ssd_hold_ms'] == 0 and injected['soft_ssd_hold_ms'] == 700 and native[key] > 0 > injected[key],
         'controlled handoff reversal evidence changed')
    phases['R10-GPU-03'] = dict(forced_pairs=7, native_deep_gain_ms=native[key], injected_handoff_deep_gain_ms=injected[key],
                               scope='forced actions and a software-injected wait; not policy gains or physical SSD throttling')

    prepared = sealed(e, 'r10-native-primary-preparation', files)
    need(prepared['lock_sha256'] == files[p_lock.name] and
         {r['block'] for r in prepared['blocks']} == {b['id'] for b in p['blocks']} and
         not ({b['id'] for b in s['blocks']} & {b['id'] for b in p['blocks']}), 'primary preparation or held-out split differs')
    phases['R10-DATA-01'] = dict(primary_blocks=len(p['blocks']), service_blocks=len(s['blocks']),
                                primary_prepared_checkpoints=prepared['exact_checkpoints'])

    result = sealed(e, 'r10-native-primary-holdout', files)
    explained = extra('r10-native-primary-action-explanation.json')
    reproduced = extra('r10-native-primary-reproduction-receipt.json')['deliveries']['primary']
    result_hash = files['r10-native-primary-holdout-summary.json']
    need(result['lock_sha256'] == files[p_lock.name] == s['primary_lock_sha256'] == explained['lock_sha256'] and
         result_hash == s['primary_summary_sha256'] == explained['primary_summary_sha256'] == reproduced['official_summary_sha256'] and
         result['archive_sha256'] == explained['archive_sha256'] == reproduced['archive_sha256'] and
         reproduced['reproduced_actions_sha256'] == files['r10-native-primary-action-explanation.json'] and
         len(result['pair_rows']) == len(p['run_pairs']) == 60 and len(explained['arms']) == 120,
         'primary run, action replay, reproduction or service dependency differs')
    r = result['result']
    phases['R10-GPU-04'] = dict(registered_pairs=60, comparable_pairs=r['successful_pairs'], point_gates=r['gates'],
                                mean_gain_ms=r['mean_gain_ms'], mean_gain_margin_ms=r['mean_gain_margin_ms'])

    diag = 'r10-native-prefill-diagnostics-v1'
    missing = available(e, [diag + '-summary.json', diag + '.zip'])
    if missing:
        pending['R10-GPU-04-supplementary'] = missing
    else:
        d = sealed(e, diag, files)
        need(len(d['pairs']) == 5 and d['arms_succeeded'] + d['arms_failed'] == 10, 'diagnostic pair coverage differs')
        phases['R10-GPU-04-supplementary'] = dict(pairs=len(d['pairs']), failed_arms=d['arms_failed'])

    cal = sealed(e, 'r10-native-service-calibration-extended-01', files)
    need(len(cal['arms']) == 13 and cal['all_anchors_found'] is True and
         s['calibration_summary_sha256'] == files['r10-native-service-calibration-extended-01-summary.json'],
         'service freeze did not use the completed baseline calibration')
    phases['R10-GPU-05-calibration'] = dict(baseline_arms=13, all_anchors_found=True)
    stream = 'r10-native-service-holdout'
    supporting = ['r10-native-service-holdout-delivery-audit.json', 'r10-native-service-result-explanation.json',
                  'r10-native-service-holdout-curves.svg', 'r10-native-service-requests.csv', 'r10-native-service-requests-audit.json',
                  'r10-native-service-reproduction-receipt.json']
    missing = available(e, [stream + '-summary.json', stream + '.zip', *supporting])
    if missing:
        pending['R10-GPU-05-single-slot'] = missing
    else:
        v = sealed(e, stream, files)
        delivery, action, request_audit = (extra(supporting[i]) for i in (0, 1, 4))
        reproduction = extra(supporting[5])['deliveries']['service']
        files.update({supporting[i]: digest(e / supporting[i]) for i in (2, 3)})
        expected = {k: v[k] for k in ('archive_sha256', 'lock_sha256')} | {'summary_sha256': files[stream + '-summary.json']}
        case_requests = {b['id']: b['requests'] for b in s['blocks']}
        need(v['lock_sha256'] == files[s_lock.name] and all(row[k] == h for row in (delivery, action, request_audit) for k, h in expected.items()) and
             request_audit['request_csv_sha256'] == files[supporting[3]] and
             request_audit['requests'] == 2 * sum(case_requests[case['block']] for case in s['cases']) and
             (reproduction['archive_sha256'], reproduction['lock_sha256'], reproduction['official_summary_sha256'],
              reproduction['reproduced_actions_sha256']) == (expected['archive_sha256'], expected['lock_sha256'], expected['summary_sha256'], files[supporting[1]]) and
             delivery['completed_pairs'] == len(s['cases']) == len(v['pairs']) == 24,
             'service summary, action replay, request CSV or archive audit differs')
        phases['R10-GPU-05-single-slot'] = dict(pairs=24, failed_arms=delivery['failed_arms'], requests=request_audit['requests'])

    contract = extra('r10-native-service-batch-contract-summary.json')['qualification']
    need(contract['existing_switch_is_policy_compatible'] is False and sealed(e, 'r10-native-two-sequence-v1', files)['passed'],
         'continuous-batching scope or two-sequence synchronous qualification changed')
    staged = 'r10-native-two-sequence-staged-v1'
    if missing := available(e, [staged + '-summary.json', staged + '.zip']):
        pending['R10-GPU-05-staged-probe'] = missing
    else:
        phases['R10-GPU-05-staged-probe'] = dict(passed=sealed(e, staged, files)['passed'], scope='correctness only')
    if include_staged_followups:
        stems = [staged, 'r10-native-two-sequence-staged-v2', 'r10-native-two-sequence-staged-v3',
                 'r10-native-two-sequence-staged-v4-shared-kv', 'r10-native-two-sequence-staged-v5-sealed-control']
        missing = available(e, [stem + ext for stem in stems for ext in ('-summary.json', '.zip')])
        if missing:
            pending['R10-GPU-05-staged-followups'] = missing
        else:
            attempts = [sealed(e, stem, files) for stem in stems]
            base = attempts[0]['run']
            for stem, attempt in zip(stems, attempts, strict=True):
                run = attempt['run']
                need(all(run[k] == base[k] for k in ('input', 'model_sha256', 'runtime_sha256', 'environment')),
                     f'{stem}: changed checkpoint, model or backend in staged follow-up')
                with ZipFile(e / (stem + '.zip')) as z:
                    need(all(member_hash(z, name) == h for name, h in run['files'].items()) and
                         json.loads(z.read('run.json')) == run, f'{stem}: changed staged raw evidence')
            need(all(not x['passed'] and x['run']['build_return_code'] != 0 for x in attempts[:2]) and
                 all(not x['passed'] and x['run']['run_return_code'] != 0 for x in attempts[2:4]),
                 'changed build or recompute control outcomes')
            control = attempts[-1]
            need(control['run']['shared_kv_stream'] is True and control['run']['reference'] == 'sealed-synchronous-B' and
                 control['passed'] is True and control['findings']['exact_logit_positions'] == 36,
                 'missing exact same-checkpoint staged transport control')
            phases['R10-GPU-05-staged-followups'] = dict(attempts=stems, exact_against_recompute=False,
                                                       exact_against_same_checkpoint=True, scope='one shared KV stream')
    return dict(scope='R10 local evidence index; not a claim that policies are superior', phases=phases, missing=pending, files_sha256=files,
                limits=dict(cross_platform='A100 or V100 has not been tested here',
                            service_continuous_batching='no compatible existing switch; policy A/B requires separate rule migration'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, default=Path('docs/evidence'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--include-staged-followups', action='store_true')
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    result = audit(a.evidence, include_staged_followups=a.include_staged_followups)
    with a.out.open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(result, indent=2, sort_keys=True), file=f)
    print(f"Evidence indexed for {len(result['phases'])} sublines; missing deliverables in {len(result['missing'])} sublines.")


if __name__ == '__main__':
    main()
