"""Cross-check sealed R10 native plans and check fail-closed catalogue parsing on CPU."""

import argparse
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path
from struct import pack
import subprocess
from uuid import uuid4


def digest(path):
    with path.open('rb') as f:
        return file_digest(f, 'sha256').hexdigest()


def run(inspector, plan, *tail, error=None):
    result = subprocess.run([str(inspector), str(plan), *map(str, tail)], capture_output=True, text=True, timeout=60)
    if error:
        assert result.returncode == 1 and error in result.stderr, (result.returncode, result.stderr)
        return None
    assert not result.returncode, (plan.name, result.returncode, result.stderr)
    return [tuple(map(int, line.split())) for line in result.stdout.splitlines()]


def verify_block(row, root, inspector):
    p = root / (row['id'] + '.plan')
    s = root / (row['id'] + '.schedule.json')
    assert digest(p) == row['wire_sha256'] and digest(s) == row['schedule_sha256'], row['id']
    trace = json.loads(s.read_bytes())['trace']
    wire = iter(p.read_text(encoding='ascii').splitlines())
    magic, block, ctx, n_obj, n_req = next(wire).split()
    block, ctx, n_obj, n_req = map(int, (block, ctx, n_obj, n_req))
    assert (magic, block, ctx, n_obj, n_req) == ('R10CAT1', 512, 3072, len(trace['objects']), len(trace['requests']))
    size = {}
    for i, obj in enumerate(trace['objects']):
        kind, raw = next(wire).split()
        size[i] = int(raw)
        assert (obj['id'], obj['tier'], obj['bytes']) == (str(i), {'K': 'RAM', 'R': 'SSD'}[kind], size[i])
    assert sum(size.values()) == row['payload_bytes']
    counts, depths, tokens = Counter(), Counter(), 0
    examples = {}
    for i, req in enumerate(trace['requests']):
        n_tok, n_options, *payload = map(int, next(wire).split())
        assert payload == [run['token_id'] for run in req['input_token_runs'] if run['count'] == 1]
        assert n_tok == len(payload) and n_options == len(req['candidates'])
        tokens += n_tok
        for option in req['candidates']:
            depth, n_dep, *deps = map(int, next(wire).split())
            assert depth == option['boundary_tokens'] and list(map(str, deps)) == option['load_order'] and n_dep == len(deps)
            if depth:
                examples.setdefault(depth, (p, i, deps, size))
            depths[depth] += 1
            counts.update(deps)
    assert next(wire, None) is None
    assert dict(depths) == {int(d): n for d, n in row['candidate_depths'].items()}
    assert sum(n > 1 for n in counts.values()) == row['referenced_from_multiple_choices']
    native = run(inspector, p)
    expected = [(n_req, n_obj, row['unique_checkpoints'], tokens, sum(depths.values()), row['referenced_from_multiple_choices']),
                *sorted(depths.items())]
    assert native == expected, (p.name, native, expected)
    return expected, examples


def host_metadata(depth, deps, sizes, *, bad=None):
    """Fabricated spans exercising compact metadata only; no model bytes or GPU runtime."""
    kv = sum(sizes[i] for i in deps[:-1])
    payload = kv + sizes[deps[-1]]
    meta = bytearray(pack('<8sIQI', b'R10META1', depth, payload + 24, 2))

    def region(source, offsets, name):
        body = pack('<QQI', source, sum(sizes[i] for i, _ in offsets), len(name)) + name
        body += pack('<I', len(offsets))
        return body + b''.join(pack('<IQQ', i, offset, sizes[i]) for i, offset in offsets)

    pieces = [(i, 0) for i in deps[:-1]]
    if bad == 'wrong-contiguous-offset':
        pieces[0] = (pieces[0][0], 4)
    if bad == 'swapped-kv-blocks':
        pieces.reverse()
    meta.extend(region(8, pieces, b'cpu_attention_fixture'))
    meta.extend(region(8 + kv - 4 if bad == 'overlapping-holes' else 16 + kv, [(deps[-1], 0)], b'cpu_state_fixture'))
    meta.extend(b'head1234gap12345tail1234')
    return bytes(meta[:-1] if bad == 'truncated-skeleton' else meta), payload


def mutations(inspector, examples, work):
    prefix = f'r10-catalogue-host-{uuid4().hex}'
    paths = []

    def write(name, data):
        loc = work / f'{prefix}-{name}'
        with loc.open('xb') as f:
            paths.append(loc)
            f.write(data)
        return loc

    try:
        return run_mutations(inspector, examples, write)
    finally:
        for path in paths:
            path.unlink()


def run_mutations(inspector, examples, write):
    outcomes = {}
    for depth, (p, req, deps, sizes) in sorted(examples.items()):
        data, size = host_metadata(depth, deps, sizes)
        loc = write(f'{depth}.meta', data)
        assert run(inspector, p, req, depth, loc) == [(depth, 2, size, 24)]
        outcomes[str(depth)] = {'expected_payload_bytes': size, 'compact_fixture_bytes': len(data)}
    depth = max(examples)
    p, req, deps, sizes = examples[depth]
    for bad, reason in {
        'wrong-contiguous-offset': 'bad R10 span object coverage',
        'swapped-kv-blocks': 'attention tensor has misordered or unequal KV blocks',
        'overlapping-holes': 'overlapping or invalid R10 metadata hole',
        'truncated-skeleton': 'bad R10 compact skeleton tail',
    }.items():
        loc = write(f'{bad}.meta', host_metadata(depth, deps, sizes, bad=bad)[0])
        run(inspector, p, req, depth, loc, error=reason)
        outcomes[bad] = 'rejected'
    invalid = write('trailing.plan', p.read_bytes() + b'unexpected\n')
    run(inspector, invalid, error='trailing content')
    outcomes['unexpected-plan-tail'] = 'rejected'
    # Use two identical 512-token requests for the two identity failure modes.
    t = '1 ' * 511
    invalid = write('changed-prefix.plan', ('R10CAT1 512 3072 2 2\nK 16777216\nR 52690944\n'
                                           f'512 2 {t}1\n0 0\n512 2 0 1\n512 2 {t}2\n0 0\n512 2 0 1\n').encode('ascii'))
    run(inspector, invalid, error='changed prefix identity')
    outcomes['conflicting-token-prefix'] = 'rejected'
    invalid = write('duplicate-identity.plan', ('R10CAT1 512 3072 3 2\nK 16777216\nR 52690944\nK 16777216\n'
                                                f'512 2 {t}1\n0 0\n512 2 0 1\n512 2 {t}1\n0 0\n512 2 2 1\n').encode('ascii'))
    run(inspector, invalid, error='split canonical identity')
    outcomes['duplicate-object-identity'] = 'rejected'
    return outcomes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('summary', 'plans', 'inspector', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args()
    manifest = json.loads(a.summary.read_bytes())
    assert len(manifest['blocks']) == 22 == len({row['id'] for row in manifest['blocks']})
    families, depths, examples = Counter(), Counter(), {}
    results = {}
    for row in manifest['blocks']:
        expected, sample = verify_block(row, a.plans, a.inspector.resolve())
        n_requests, n_obj, n_checkpoints, n_tokens, n_candidates, shared = expected[0]
        results[row['id']] = dict(requests=n_requests, objects=n_obj, checkpoints=n_checkpoints,
                                  tokens=n_tokens, candidates=n_candidates, shared_object_ids=shared)
        examples.update(sample)
        families[row['family']] += 1
        depths.update(dict(expected[1:]))
    assert set(examples) == {512, 1024, 1536}
    fixture = mutations(a.inspector.resolve(), examples, a.plans)
    evidence = dict(scope='CPU parser parity and fabricated metadata rejection; no GPU or storage performance measurements',
                    native_plan_manifest_sha256=digest(a.summary), families=dict(families), depths=dict(depths),
                    checked_blocks=results, host_fixture=fixture)
    with a.out.open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True), file=f)
    print(f'Checked {len(results)} source blocks, {sum(depths.values())} candidates and seven parser rejection cases.')


if __name__ == '__main__':
    main()
