"""Rebuild the R10 holdout results from the frozen plans and archives, without a GPU."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import re
from shutil import copyfileobj
import subprocess
import sys
from zipfile import ZipFile

from scripts.validation.check_r10_native_catalog import digest


EVIDENCE = Path('docs/evidence')
INPUTS = Path('outputs/r10/local-reanalysis-inputs-v1.zip')
COMPONENT = re.compile(r'^[A-Za-z0-9_][A-Za-z0-9_.-]*[A-Za-z0-9_-]$')


def read(path):
    return json.loads(path.read_bytes())


def need(condition, message):
    if not condition:
        raise ValueError(message)


def members(archive):
    records = archive.infolist()
    names = [rec.filename for rec in records]
    need(len({name.casefold() for name in names}) == len(names), 'duplicate or case-conflicting archive paths')
    for rec in records:
        need(not rec.is_dir() and all(COMPONENT.fullmatch(part) for part in rec.filename.split('/')),
             f'invalid output path in archive: {rec.filename}')
    return records


def unpack(path, output, expected_hash, manifest=None):
    need(digest(path) == expected_hash, f'archive SHA-256 mismatch: {path}')
    with ZipFile(path) as archive:
        records = members(archive)
        if manifest is not None:
            files = manifest['files']
            need({rec.filename for rec in records} == files.keys() | {'manifest.json'}, 'unexpected input archive members')
            need(json.loads(archive.read('manifest.json')) == {k: v for k, v in manifest.items() if k != 'archive_sha256'},
                 'input archive and published manifest disagree')
        output.mkdir()
        for rec in records:
            target = output.joinpath(*rec.filename.split('/'))
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(rec) as src, target.open('xb') as dst:
                copyfileobj(src, dst)
            if manifest is not None and rec.filename != 'manifest.json':
                expected = files[rec.filename]
                need((target.stat().st_size, digest(target)) == (expected['bytes'], expected['sha256']),
                     f'input member differs from manifest: {rec.filename}')
    return len(records)


def run(name, **options):
    cmd = [sys.executable, '-B', '-m', 'scripts.gpu.analysis.' + name]
    for key, value in options.items():
        cmd += ['--' + key.replace('_', '-'), str(value)]
    subprocess.run(cmd, check=True)


def compare_json(actual, expected, *, archive_hash=False):
    observed, official = read(actual), read(expected)
    observed, official = (dict(d) for d in (observed, official))
    if archive_hash:
        # ZIP encoding can vary by zlib build; raw content is checked separately.
        for d in (observed, official):
            d.pop('archive_sha256', None)
    need(observed == official, f'reproduced JSON differs from official evidence: {expected.name}')


def member_hashes(path):
    with ZipFile(path) as z:
        result = {}
        for rec in members(z):
            with z.open(rec) as f:
                result[rec.filename] = file_digest(f, 'sha256').hexdigest()
        return result


def reproduce(a, mode, inputs):
    e = a.evidence
    raw = e / f'r10-native-{mode}-holdout.zip'
    official = raw.with_name(raw.stem + '-summary.json')
    lock_stem = 'r10-local-holdout' if mode == 'primary' else 'r10-local-service-holdout'
    lock = e / (lock_stem + '-execution-lock.json')
    original_summary = read(official)
    need(original_summary['lock_sha256'] == digest(lock), f'{mode} formal summary belongs to a different lock')
    if mode == 'primary':
        need((inputs / 'settings' / lock.name).read_bytes() == lock.read_bytes(), 'input pack contains another primary lock')
    replay = a.out / mode
    replay.mkdir()
    campaign = replay / 'campaign'
    n = unpack(raw, campaign, original_summary['archive_sha256'])
    need((campaign / 'execution-lock.json').read_bytes() == lock.read_bytes(), f'{mode} archive contains a different lock')
    earned = replay / 'summary.json'
    plans = inputs / 'plans'
    extra = inputs / 'service-plans'
    common = dict(lock=lock, plans=plans, campaign=campaign, out=earned)
    cost = inputs / 'settings/r10-local-holdout-cost-v1.json'
    action = replay / 'action-explanation.json'
    if mode == 'primary':
        run('summarize_r10_holdout', **common)
        compare_json(earned, official, archive_hash=True)
        run('explain_r10_primary_actions', lock=lock, plans=plans, cost=cost, archive=raw, primary_summary=official,
            preparation_archive=e / 'r10-native-primary-preparation.zip',
            preparation_summary=e / 'r10-native-primary-preparation-summary.json', out=action)
        compare_json(action, e / 'r10-native-primary-action-explanation.json')
    else:
        derived_zip = replay / 'reproduced-archive.zip'
        run('summarize_r10_service_holdout', extra_plans=extra, archive=derived_zip, **common)
        compare_json(earned, official, archive_hash=True)
        need(member_hashes(derived_zip) == member_hashes(raw), 'reproduced service archive members differ')
        run('explain_r10_service_results', lock=lock, summary=official, archive=raw, plans=plans, extra_plans=extra,
            delivery=e / 'r10-native-service-holdout-delivery-audit.json', cost=cost,
            calibration_design=inputs / 'settings/r10-service-calibration-v1.json', out=action)
        compare_json(action, e / 'r10-native-service-result-explanation.json')
        svg = replay / 'curves.svg'
        run('plot_r10_service_curves', summary=earned, out=svg)
        need(svg.read_bytes() == (e / 'r10-native-service-holdout-curves.svg').read_bytes(), 'reproduced service curves differ')
    print(f'{mode}: formal summary and action explanation reproduced from {n} archive members.', flush=True)
    return dict(archive_sha256=original_summary['archive_sha256'], lock_sha256=digest(lock), members=n,
                official_summary_sha256=digest(official), reproduced_summary_sha256=digest(earned), reproduced_actions_sha256=digest(action))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, default=EVIDENCE)
    p.add_argument('--inputs', type=Path, default=INPUTS)
    p.add_argument('--out', type=Path, required=True, help='new local directory; extracted plans include prompt tokens')
    p.add_argument('--mode', choices=('primary', 'service', 'both'), required=True)
    a = p.parse_args()
    modes = ('primary', 'service') if a.mode == 'both' else (a.mode,)
    for mode in modes:
        stem = a.evidence / f'r10-native-{mode}-holdout'
        need(all(stem.with_name(stem.name + s).is_file() for s in ('-summary.json', '.zip')), f'{mode} delivery is not present')
    a.out.mkdir()
    manifest = read(a.evidence / 'r10-local-reanalysis-inputs-v1-summary.json')
    inputs = a.out / 'inputs'
    n = unpack(a.inputs, inputs, manifest['archive_sha256'], manifest)
    deliveries = {mode: reproduce(a, mode, inputs) for mode in modes}
    with (a.out / 'reproduction-receipt.json').open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(dict(scope='archive-only analysis, no model or accelerator execution', input_members=n,
                              input_archive_sha256=manifest['archive_sha256'], deliveries=deliveries), indent=2, sort_keys=True), file=f)


if __name__ == '__main__':
    main()
