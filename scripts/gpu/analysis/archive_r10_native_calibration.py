"""Collect the eight native preparation receipts without archiving prompt or tensor payloads."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from scripts.validation.check_r10_native_catalog import digest


def collect(manifest, root):
    cuts = {r['id']: r for r in json.loads(manifest.read_bytes())['blocks'] if r['split'] in ('calibration', 'pilot')}
    entries, found, depths, scope = {}, {}, Counter(), set()
    for sealed in sorted(root.glob('*.seal.json')):
        packet = json.loads(sealed.read_bytes())
        block = packet['block']
        assert block in cuts and block not in found, block
        stem = sealed.name.removesuffix('.seal.json')
        side = {key: root / f'{stem}.{key}' for key in ('csv', 'stderr', 'run.json')}
        receipt = json.loads(side['run.json'].read_bytes())
        assert receipt['status'] == 'exited' and receipt['return_code'] == 0 and receipt['block'] == block
        assert digest(side['run.json']) == packet['run_receipt_sha256']
        assert digest(side['csv']) == packet['producer_csv_sha256']
        assert digest(side['stderr']) == packet['producer_stderr_sha256']
        assert cuts[block]['schedule_sha256'] == packet['schedule_sha256']
        assert cuts[block]['wire_sha256'] == receipt['plan_sha256'] == packet['wire_sha256']
        cmd = dict(model_sha256=receipt['model_sha256'], dll_sha256=receipt['dll_sha256'], device=receipt['device_name'])
        scope.add(json.dumps(cmd, sort_keys=True))
        with side['csv'].open(encoding='ascii', newline='') as f:
            rows = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(f)]
        assert len(rows) == len(packet['checkpoints']) == cuts[block]['unique_checkpoints']
        assert {str(row['checkpoint']) for row in rows} == set(packet['checkpoints'])
        assert all((row['exact_state'], row['exact_logits_positions'], row['exact_final_state']) == (1, 8, 1) for row in rows)
        d = Counter(row['depth'] for row in rows)
        depths.update(d)
        found[block] = dict(family=cuts[block]['family'], requests=cuts[block]['requests'], checkpoints=len(rows),
                            depth_counts=dict(d), shared_kv_comparisons=sum(r['shared_identity_checks'] for r in rows),
                            continuation_positions=8 * len(rows), payload_bytes=sum(o['bytes'] for o in packet['objects'].values()),
                            probe_sha256=receipt['probe_sha256'], receipt_sha256=packet['run_receipt_sha256'],
                            seal_sha256=digest(sealed), qualification_seconds=receipt['duration_s'])
        for file in [sealed, *side.values()]:
            entries[f'{block}/{file.name}'] = file.read_bytes()
    assert set(found) == set(cuts) and len(scope) == 1 and set(depths) == {512, 1024, 1536}
    disk = root / 'overlapped-disk-probe.csv'
    with disk.open(encoding='ascii', newline='') as f:
        disk_rows = list(csv.DictReader(f))
    assert len(disk_rows) == 3 and all(r['bytes'] == '52690944' and r['content_equal'] == '1' for r in disk_rows)
    entries[disk.name] = disk.read_bytes()
    return dict(scope='untimed native catalogue qualification; no timed A/B or holdout', blocks=found,
                runtime=json.loads(scope.pop()), depth_counts=dict(depths), checkpoints=sum(depths.values()),
                continuation_positions=8 * sum(depths.values()), shared_kv_comparisons=sum(r['shared_kv_comparisons'] for r in found.values()),
                disk_probe=dict(objects=len(disk_rows), successful_bytewise_comparisons=len(disk_rows),
                                observed_pending=sum(int(r['incomplete_polls']) > 0 for r in disk_rows))), entries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'root', 'archive', 'summary'):
        ap.add_argument('--' + name, required=True, type=Path)
    a = ap.parse_args()
    if a.archive.exists() or a.summary.exists():
        raise FileExistsError('choose new archive and summary names')
    report, raw = collect(a.manifest, a.root)
    with ZipFile(a.archive, 'x') as z:
        for name, contents in sorted(raw.items()):
            z.writestr(ZipInfo(name), contents, compress_type=ZIP_DEFLATED)
    report['archive_sha256'] = digest(a.archive)
    with a.summary.open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), file=f)
    print(f"Archived {report['checkpoints']} checkpoints with {report['continuation_positions']} exact continuation positions.")


if __name__ == '__main__':
    main()
