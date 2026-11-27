"""Archive only completed native state fixtures that hit their required physical paths."""

import argparse
import csv
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from scripts.gpu.analysis.check_r10_catalog_fifo import check
from scripts.validation.check_r10_native_catalog import digest


def collect(root):
    manifest = root / 'manifest.json'
    source = json.loads(manifest.read_bytes())
    entries = {'manifest.json': manifest.read_bytes()}
    output = {}
    for block in source['blocks']:
        ident = block['id']
        prefix = root / (ident + '-verified-v1')
        run, native, errs = [prefix.with_suffix(ext) for ext in ('.run.json', '.native.csv', '.stderr')]
        schedule = root / 'plans' / (ident + '.schedule.json')
        assert digest(schedule) == block['schedule_sha256']
        report = check(run, native, schedule)
        run_data = json.loads(run.read_bytes())
        counts = report['device_event_counts']
        assert run_data['qualification'] and counts['exact-state'] == counts['exact-final-logits'] == block['requests']
        assert run_data['capacity_bytes'] == 69468160 and report['peak_prefetch_bytes'] == run_data['capacity_bytes']
        with native.open(encoding='ascii', newline='') as f:
            events = list(csv.DictReader(f))

        def select(event, request):
            return [i for i, row in enumerate(events) if row['event'] == event and row['request'] == str(request)]

        if ident == 'mixed-private-staged':
            assert run_data['publish_rest_on'] == 'first-compute' and report['direct_plus_scatter_requests'] == [1]
            reads = [events[i] for i in select('read-direct', 1)]
            assert len(reads) == 2 and len(select('native-scatter', 1)) == 1, reads
        elif ident in ('direct-handoff', 'nonadjacent-loan'):
            assert run_data['publish_rest_on'] == 'first-ssd'
            borrower = 1 if ident == 'direct-handoff' else 2
            [start, recurring] = select('read-direct', 0)
            assert start < recurring < select('reserved', borrower)[0] < select('upload-ready', 0)[1]
            assert len(select('preserve-recurrent', 0)) == 1
            if borrower == 1:
                assert len(select('carry-active-kv', 1)) == 1
                assert counts.get('preserve-old-active', 0) == 0
            else:
                assert len(select('preserve-old-active', 1)) == 1 and len(select('native-scatter', 2)) == 2
        else:
            raise ValueError(f'unknown qualification fixture {ident}')
        packet = root / f'{ident}-native.seal.json'
        assert run_data['seal_sha256'] == digest(packet)
        report['publish_rest_on'] = run_data['publish_rest_on']
        report['catalogue_seal_sha256'] = digest(packet)
        output[ident] = report
        for file in (run, native, errs, packet):
            entries[f'{ident}/{file.name}'] = file.read_bytes()
    assert set(output) == {'mixed-private-staged', 'direct-handoff', 'nonadjacent-loan'}
    return dict(scope='event-anchored native path correctness, excluding timings', fixtures=output,
                full_state_matches=sum(x['device_event_counts']['exact-state'] for x in output.values()),
                final_logits_matches=sum(x['device_event_counts']['exact-final-logits'] for x in output.values())), entries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ('root', 'archive', 'summary'):
        ap.add_argument('--' + arg, required=True, type=Path)
    a = ap.parse_args()
    if a.archive.exists() or a.summary.exists():
        raise FileExistsError('select new paths')
    summary, entries = collect(a.root)
    with ZipFile(a.archive, 'x') as z:
        for name, contents in sorted(entries.items()):
            z.writestr(ZipInfo(name), contents, compress_type=ZIP_DEFLATED)
    summary['archive_sha256'] = digest(a.archive)
    with a.summary.open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2), file=f)
    print(f"Archived {summary['full_state_matches']} state matches over {len(summary['fixtures'])} confirmed native paths.")


if __name__ == '__main__':
    main()
