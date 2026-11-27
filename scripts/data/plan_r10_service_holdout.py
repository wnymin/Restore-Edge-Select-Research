"""Freeze disjoint service input identities; do not prepare native state or run requests."""

import argparse
import json
from pathlib import Path

from scripts.data.audit_r10_oasst import audit
from scripts.data.build_r10_native_plans import digest, load_oasst, write_plan


SCOPE = 'R10 service held-out membership only; no execution authorization'


def read(path):
    return json.loads(path.read_bytes())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('main-layout', 'native-manifest', 'bailian-manifest', 'oasst-original', 'oasst-extended',
                 'oasst-tokens', 'extra-plans', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.out.exists() or a.extra_plans.exists():
        raise FileExistsError('choose new paths for independent service membership and offline plans')
    layout = read(a.main_layout)
    assert layout['source_manifests']['native_sha256'] == digest(a.native_manifest)
    assert layout['source_manifests']['bailian_sha256'] == digest(a.bailian_manifest)
    taken = {r['id'] for r in layout['blocks']}
    ext, original = map(read, (a.oasst_extended, a.oasst_original))
    assert len(original['blocks']) == 10 and ext['blocks'][:10] == original['blocks']
    assert all(ext[key] == value for key, value in original.items() if key not in ('blocks', 'private_tokens_sha256'))
    oasst_audit = audit(a.oasst_extended, a.oasst_tokens)
    assert oasst_audit['request_counts']['service-holdout'] == 128
    trace_specs = {Path(x['generated_filename']).stem: x for x in read(a.bailian_manifest)['cases']}
    bailian = []
    for row in read(a.native_manifest)['blocks']:
        if row['family'] != 'bailian' or row['split'] != 'reserved' or row['id'] in taken:
            continue
        spec = trace_specs[row['id']]
        assert row['source_sha256'] == spec['generated_sha256']
        for old in (x for x in layout['blocks'] if x.get('source') == spec['source']):
            a0, a1 = spec['raw_rows']
            b0, b1 = old['raw_rows']
            assert a1 <= b0 or b1 <= a0
        bailian.append(dict(row, source=spec['source'], raw_rows=spec['raw_rows'], raw_window_sha256=spec['raw_window_sha256']))
    assert len(bailian) == 2 and len({r['source'] for r in bailian}) == 2
    service = [s for s in load_oasst(a.oasst_extended, a.oasst_tokens) if s['split'] == 'service-holdout']
    assert [s['id'] for s in service] == ['oasst-service-hold-01', 'oasst-service-hold-02']
    assert taken.isdisjoint(s['id'] for s in service)
    a.extra_plans.mkdir(parents=True, exist_ok=False)
    oasst = [write_plan(s, a.extra_plans) for s in service]
    proof = dict(scope=SCOPE, blocks=sorted(bailian + oasst, key=lambda x: x['id']),
                 main_layout_sha256=digest(a.main_layout), native_manifest_sha256=digest(a.native_manifest),
                 oasst_original_sha256=digest(a.oasst_original), oasst_verified=oasst_audit)
    with a.out.open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(proof, sort_keys=True, indent=2), file=f)


if __name__ == '__main__':
    main()
