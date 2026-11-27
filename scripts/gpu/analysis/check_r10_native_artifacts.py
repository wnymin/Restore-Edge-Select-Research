"""Seal a complete R10 GPU catalogue against its plan and exact-restore records."""

import argparse
import csv
import io
import json
from pathlib import Path
from struct import Struct

from scripts.validation.check_r10_native_catalog import digest


HEADER = ('checkpoint', 'depth', 'dependencies', 'full_bytes', 'metadata_bytes', 'shared_identity_checks',
          'preparation_us', 'exact_state', 'exact_logits_positions', 'exact_final_state')
META_HEADER = Struct('<8sIQI')
SPAN = Struct('<QQI')
PART = Struct('<IQQ')
U32 = Struct('<I')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def inspect_metadata(path, depth, deps, size, image):
    raw = path.read_bytes()
    pos = 0

    def pull(shape):
        nonlocal pos
        need(pos + shape.size <= len(raw), f'{path.name}: truncated descriptor')
        data = shape.unpack_from(raw, pos)
        pos += shape.size
        return data

    magic, d, full, spans = pull(META_HEADER)
    payload = sum(size[o] for o in deps)
    need((magic, d, full) == (b'R10META1', depth, image) and 0 < spans <= 512 and 0 < full - payload < (2 << 20),
         f'{path.name}: descriptor/header mismatch')
    storage = {o: 0 for o in deps}
    holes = []
    for _ in range(spans):
        source, n, name_len = pull(SPAN)
        need(0 < name_len < 64 and pos + name_len <= len(raw), f'{path.name}: invalid native tensor name')
        name = raw[pos:pos + name_len]
        # ggml gives some native tensor views a literal " (view)" suffix.
        need(bool(name.strip()) and all(32 <= c < 127 for c in name), f'{path.name}: invalid native tensor name bytes')
        pos += name_len
        count, = pull(U32)
        need(0 < count < len(deps), f'{path.name}: invalid number of parts')
        parts = []
        for _ in range(count):
            obj, start, length = pull(PART)
            need(obj in storage and storage[obj] == start and 0 < length <= size[obj] - start and length % 4 == 0,
                 f'{path.name}: invalid or repeated object bytes')
            storage[obj] += length
            parts.append((obj, length))
        need(sum(length for _, length in parts) == n, f'{path.name}: partial span')
        ids = tuple(obj for obj, _ in parts)
        need(ids == deps[-1:] or (ids == deps[:-1] and len({length for _, length in parts}) == 1),
             f'{path.name}: unexpected object ordering within native tensor')
        holes.append((source, n))
    need(storage == {o: size[o] for o in deps}, f'{path.name}: partial object')
    previous = 0
    for source, n in sorted(holes):
        need(previous <= source and source + n <= full, f'{path.name}: overlapping or excessive zeroed range')
        previous = source + n
    need(len(raw) == pos + full - payload, f'{path.name}: excess or incomplete non-tensor bytes')
    return len(raw)


def seal(manifest, block, plans, directory, transcript, transcript_errors):
    source = json.loads(manifest.read_bytes())
    found = [b for b in source['blocks'] if b['id'] == block]
    need(len(found) == 1, f'{block}: one manifest entry required')
    row = found[0]
    wire, schedule = (plans / (block + ext) for ext in ('.plan', '.schedule.json'))
    wire_hash, schedule_hash = digest(wire), digest(schedule)
    need(wire_hash == row['wire_sha256'] and schedule_hash == row['schedule_sha256'], f'{block}: changed input files')
    trace = json.loads(schedule.read_bytes())['trace']
    sizes = {int(o['id']): o['bytes'] for o in trace['objects']}
    checkpoints = {}
    for req in trace['requests']:
        for option in req['candidates']:
            depth, deps = option['boundary_tokens'], tuple(map(int, option['load_order']))
            if depth:
                ident = deps[-1]
                need(checkpoints.setdefault(ident, (depth, deps)) == (depth, deps), 'conflicting recurrent identity')
    need(len(checkpoints) == row['unique_checkpoints'], 'incorrect checkpoint count')
    need(set(directory.iterdir()) == ({directory / f'{i}.bin' for i in sizes} |
                                    {directory / f'{i}.meta' for i in checkpoints}),
         f'{block}: partial or foreign catalogue files')
    data = transcript.read_bytes()
    need(data.endswith(b'\n'), f'{block}: transcript has an incomplete last row')
    rows = csv.DictReader(io.StringIO(data.decode('ascii')))
    need(tuple(rows.fieldnames or ()) == HEADER, f'{block}: invalid transcript header')
    output = {}
    used = set()
    order = sorted(checkpoints)
    for i, record in enumerate(rows):
        need(i < len(order) and set(record) == set(HEADER), 'unexpected or extra transcript column/row')
        record = {k: int(v) for k, v in record.items()}
        ident = record['checkpoint']
        need(ident == order[i], 'duplicated, missing or reordered checkpoint')
        depth, deps = checkpoints[ident]
        keys = ('depth', 'dependencies', 'shared_identity_checks', 'exact_state', 'exact_logits_positions', 'exact_final_state')
        need(tuple(record[k] for k in keys) == (depth, len(deps), len(used.intersection(deps)), 1, 8, 1) and
             record['preparation_us'] >= 0, f'{ident}: exact state / decoded continuation did not pass or reference counters differ')
        used.update(deps)
        desc = directory / f'{ident}.meta'
        actual = inspect_metadata(desc, depth, deps, sizes, record['full_bytes'])
        need(actual == record['metadata_bytes'], f'{ident}: transcript metadata lengths differ from files')
        output[str(ident)] = dict(depth=depth, dependencies=deps, state_bytes=record['full_bytes'],
                                  exact_logits_positions=record['exact_logits_positions'], metadata_bytes=actual,
                                  metadata_sha256=digest(desc), preparation_us=record['preparation_us'])
    need(len(output) == len(checkpoints) and used == set(sizes), f'{block}: incomplete exact-restore transcript')
    objects = {}
    for ident, n in sorted(sizes.items()):
        raw = directory / f'{ident}.bin'
        need(raw.stat().st_size == n, f'{ident}: incorrect object payload length')
        objects[str(ident)] = dict(bytes=n, sha256=digest(raw))
    return dict(scope='untimed GPU catalogue preparation; this seal does not certify online execution or performance',
                block=block, split=row['split'], family=row['family'], wire_sha256=wire_hash, schedule_sha256=schedule_hash,
                producer_csv_sha256=digest(transcript), producer_stderr_sha256=digest(transcript_errors),
                checkpoints=output, objects=objects)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--block', required=True)
    for name in ('manifest', 'plans', 'catalogue', 'transcript', 'transcript-errors', 'run-receipt', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args()
    receipt = json.loads(a.run_receipt.read_bytes())
    need(receipt['status'] == 'exited' and receipt['return_code'] == 0 and receipt['block'] == a.block,
         'only a complete, successful native process can be sealed')
    result = seal(a.manifest, a.block, a.plans, a.catalogue, a.transcript, a.transcript_errors)
    need(receipt['plan_sha256'] == result['wire_sha256'], 'receipt refers to a different plan')
    result['run_receipt_sha256'] = digest(a.run_receipt)
    with a.out.open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2), file=f)
    print(f"Sealed {len(result['checkpoints'])} native checkpoints in {a.block}.")


if __name__ == '__main__':
    main()
