"""Pack the frozen model inputs needed to recalculate R10 results on this machine."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from scripts.gpu.experiments.r10_holdout_lock import load
from scripts.validation.check_r10_native_catalog import digest


def read(path):
    return json.loads(path.read_bytes())


def members(a):
    lock, service, design = load(a.primary_lock), read(a.service_layout), read(a.service_design)
    layout_hash, native_hash = digest(a.primary_layout), digest(a.native_manifest)
    if (lock['layout_sha256'], lock['native_manifest_sha256'], lock['cost_profile_sha256']) != (
            layout_hash, native_hash, digest(a.cost_profile)) or (service['main_layout_sha256'], service['native_manifest_sha256']) != (layout_hash, native_hash):
        raise ValueError('input inventories and locked primary configuration do not match')
    by_id = {b['id']: b for b in read(a.native_manifest)['blocks']}
    service_calibration = [by_id[ident] for ident in design['block_order']]
    if len(service_calibration) != 2 or any((b['family'], b['schedule_sha256']) != (
            design['blocks'][b['id']]['family'], design['blocks'][b['id']]['schedule_sha256']) or
            b['split'] not in ('calibration', 'pilot') for b in service_calibration):
        raise ValueError('service calibration plans do not match the declared design')
    inventories = {'primary': lock['blocks'], 'service-calibration': service_calibration, 'service-holdout': service['blocks']}
    ids = [b['id'] for items in inventories.values() for b in items]
    if len(ids) != len(set(ids)) or len(service['blocks']) != 4 or sorted(b['family'] for b in service['blocks']) != ['bailian'] * 2 + ['oasst'] * 2:
        raise ValueError('input bundle requires distinct blocks in the declared scopes')
    result = {}
    for name in ('primary_lock', 'primary_layout', 'native_manifest', 'service_layout', 'service_design', 'cost_profile', 'service_calibration_cost'):
        path = getattr(a, name)
        result['settings/' + path.name] = (path, digest(path))
    for stage, blocks in inventories.items():
        for b in blocks:
            ident = b['id']
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', ident):
                raise ValueError(f'invalid plan identifier: {ident}')
            extra = stage == 'service-holdout' and b['family'] == 'oasst'
            for suffix, hash_key in (('.plan', 'wire_sha256'), ('.schedule.json', 'schedule_sha256')):
                name = ident + suffix
                result[('service-plans/' if extra else 'plans/') + name] = ((a.extra_plans if extra else a.plans) / name, b[hash_key])
    return result, {stage: [b['id'] for b in blocks] for stage, blocks in inventories.items()}


def manifest(entries, inventories):
    return dict(scope='local R10 reanalysis inputs, no result logs, application, model weights or native tensor payloads',
                blocks=inventories, files={name: dict(bytes=path.stat().st_size, sha256=expected)
                                           for name, (path, expected) in sorted(entries.items())})


def put(z, name, raw):
    z.writestr(ZipInfo(name), raw, compress_type=ZIP_DEFLATED, compresslevel=1)


def pack(entries, index, archive):
    # Verify input bytes before producing the archive. Small inputs are separate from raw tensor payloads.
    for name, (path, expected) in entries.items():
        if digest(path) != expected:
            raise ValueError(f'input bytes differ from the freeze: {name}')
    with ZipFile(archive, 'x') as z:
        for name, (path, _) in sorted(entries.items()):
            put(z, name, path.read_bytes())
        put(z, 'manifest.json', (json.dumps(index, indent=2, sort_keys=True) + '\n').encode('utf-8'))
    with ZipFile(archive) as z:
        if set(z.namelist()) != set(entries) | {'manifest.json'} or len(z.namelist()) != len(entries) + 1:
            raise ValueError('input archive members differ from the intended set')
        if json.loads(z.read('manifest.json')) != index:
            raise ValueError('input archive manifest is inconsistent')
        for name, record in index['files'].items():
            raw = z.read(name)
            if (len(raw), sha256(raw).hexdigest()) != (record['bytes'], record['sha256']):
                raise ValueError(f'archived input failed readback: {name}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('primary-lock', 'primary-layout', 'native-manifest', 'service-layout', 'service-design', 'cost-profile',
                 'service-calibration-cost', 'plans', 'extra-plans', 'archive', 'report'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--dry-run', action='store_true', help='verify configuration and file presence without reading or writing plan payloads')
    a = p.parse_args()
    if not a.archive.resolve().is_relative_to(Path('outputs').resolve()) or a.archive.exists() or a.report.exists():
        p.error('select a fresh archive under ignored outputs/ and a fresh summary')
    entries, inventories = members(a)
    index = manifest(entries, inventories)
    if a.dry_run:
        print(json.dumps(dict(block_counts={k: len(v) for k, v in inventories.items()}, files=len(entries),
                              uncompressed_bytes=sum(x['bytes'] for x in index['files'].values()))))
        return
    pack(entries, index, a.archive)
    with a.report.open('x', encoding='utf-8', newline='\n') as f:
        print(json.dumps(dict(index, archive_sha256=digest(a.archive)), sort_keys=True, indent=2), file=f)


if __name__ == '__main__':
    main()
