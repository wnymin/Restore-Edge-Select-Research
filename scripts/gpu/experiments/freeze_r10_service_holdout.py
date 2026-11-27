"""Freeze local streaming cases after baseline calibration and primary completion."""

import argparse
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

from scripts.data.plan_r10_service_holdout import SCOPE as MEMBERSHIP
from scripts.gpu.experiments import r10_holdout_lock as primary, r10_service_lock as service
from scripts.gpu.experiments.run_r10_catalog_prepare import write_new
from scripts.gpu.experiments.run_r10_service_calibration import EXTENSION_SCOPE, generation_shape, matrix, service_deadline
from scripts.gpu.analysis.summarize_r10_service_calibration import anchors
from scripts.validation.check_r10_native_catalog import digest


def read(path):
    return json.loads(path.read_bytes())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('membership', 'primary-lock', 'primary-summary', 'design', 'calibration-summary', 'calibration-archive',
                'calibration-stamp', 'worker', 'cost-profile', 'plans', 'extra-plans', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    base = primary.load(a.primary_lock)
    primary.check_controller(base)
    if digest(a.cost_profile) != base['cost_profile_sha256']:
        raise ValueError('hold the stationary cost predictor fixed across the two phases')
    base_summary = read(a.primary_summary)
    assert base_summary['lock_sha256'] == digest(a.primary_lock)
    assert [(r['block'], r['round']) for r in base_summary['pair_rows']] == [(r['block'], r['round']) for r in base['run_pairs']]
    layout, design, cal, stamp = map(read, (a.membership, a.design, a.calibration_summary, a.calibration_stamp))
    assert layout['scope'] == MEMBERSHIP and layout['main_layout_sha256'] == base['layout_sha256']
    assert {b['id'] for b in layout['blocks']}.isdisjoint(b['id'] for b in base['blocks'])
    assert cal['all_anchors_found'] and cal['archive_sha256'] == digest(a.calibration_archive)
    assert cal['design_sha256'] == stamp['design_sha256'] == digest(a.design)
    added = []
    with ZipFile(a.calibration_archive) as z:
        for i, expected in enumerate(cal['extensions_sha256'], 1):
            raw = z.read(f'extension/{i:02d}.json')
            ext = json.loads(raw)
            assert sha256(raw).hexdigest() == expected and ext['scope'] == EXTENSION_SCOPE
            assert ext['design_sha256'] == cal['design_sha256']
            added.extend(ext['points'])
    expected = [(b, p, r, Fraction(details['arrival_scale'][r])) for b, p, r, details, _ in matrix(design, added)]
    actual = [(r['block'], r['profile'], r['rate'], Fraction(r['arrival_scale'])) for r in cal['arms']]
    if expected != actual or len(set(actual)) != len(actual):
        raise ValueError('service anchor evidence must retain all original calibration arms and registered additions')
    for row, info in design['blocks'].items():
        for profile in service.PROFILES:
            group = [a for a in cal['arms'] if (a['block'], a['family'], a['profile']) == (row, info['family'], profile)]
            found = anchors(group, design['anchor_rules'])
            assert all(found.values()) and found == cal['chosen'][f'{info["family"]}/{profile}']
    assert stamp['controller_sha256'] == primary.controller_hashes() and stamp['runtime_sha256'] == base['environment']['runtime_sha256']
    for key in ('model', 'observer'):
        assert stamp['sha256'][key] == base['environment'][key + '_sha256']
    assert stamp['sha256']['worker'] == digest(a.worker) and stamp['sha256']['cost_profile'] == read(a.cost_profile)['calibration_cost_sha256']
    assert all(stamp['service_sha256'].get(key) == value for key, value in service.source_hashes().items())
    fixed = base['timing']['arrival_scale']
    coder = Fraction(design['holdout_rate_transfer']['bailian_reference_prefill_scale'])
    assert Fraction(fixed['qwen_coder_blksz_16.jsonl']) == coder
    seed = design['service_pair_order_seed']
    def order(label):
        return sha256(f'{seed}/{label}'.encode('ascii')).hexdigest()
    cases = []
    deadlines = []
    for b in sorted(layout['blocks'], key=lambda row: (row['family'], order(row['id']))):
        schedule = (a.extra_plans if b['family'] == 'oasst' else a.plans) / (b['id'] + '.schedule.json')
        if digest(schedule) != b['schedule_sha256']:
            raise ValueError('static service schedule changed before freeze')
        data = read(schedule)
        source_rows = data['trace']['requests']
        for profile in service.PROFILES:
            caps = design['output_profiles'][profile]
            generation_shape(data, caps)
            for anchor in service.ANCHORS:
                chosen = cal['chosen'][f'{b["family"]}/{profile}'][anchor]
                options = [r for r in cal['arms'] if (r['family'], r['profile'], r['rate']) == (b['family'], profile, chosen)]
                assert len(options) == 1 and not options[0]['failed']
                scale = Fraction(options[0]['arrival_scale'])
                if b['family'] == 'bailian':
                    scale = scale * Fraction(fixed[b['source']]) / coder
                deadlines.append(service_deadline(design, source_rows, scale))
                cases.append(dict(id=f'{b["id"]}-{profile}-{anchor}', block=b['id'], profile=profile, anchor=anchor,
                                  arrival_scale=str(scale), short_tokens=caps['short_tokens'], long_tokens=caps['long_tokens']))
    # Counterbalance so each input-family/profile/anchor combination has both orders.
    by_case = {r['id']: r for r in cases}
    for family in ('oasst', 'bailian'):
        family_blocks = {b['id'] for b in layout['blocks'] if b['family'] == family}
        for profile in service.PROFILES:
            for anchor in service.ANCHORS:
                ids = sorted((r['id'] for r in cases if r['block'] in family_blocks and r['profile'] == profile and r['anchor'] == anchor), key=order)
                for i, ident in enumerate(ids):
                    by_case[ident]['arm_order'] = list(primary.POLICIES[::1 if i == 0 else -1])
    cases.sort(key=lambda r: order(r['id']))
    write_new(a.out, dict(scope=service.SCOPE, membership_sha256=digest(a.membership), primary_lock_sha256=digest(a.primary_lock),
                          primary_summary_sha256=digest(a.primary_summary), calibration_summary_sha256=digest(a.calibration_summary),
                          calibration_design_sha256=digest(a.design), blocks=layout['blocks'], cases=cases,
                          environment=dict(base['environment'], worker_sha256=digest(a.worker), service_sha256=service.source_hashes(),
                                           service_analysis_sha256=service.analysis_hashes()),
                          cost_profile_sha256=base['cost_profile_sha256'], experimental_slo=design['experimental_slo'],
                          capacity_mib=design['capacity_mib'], timeout_s=max(deadlines)))
    service.load(a.out)


if __name__ == '__main__':
    main()
