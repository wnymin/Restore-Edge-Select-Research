"""Translate sealed Bailian and OASST inputs to a shared native object catalogue."""

import argparse
from collections import Counter
from fractions import Fraction
import gzip
from hashlib import sha256, file_digest
import json
from pathlib import Path
from struct import pack

from restore_edge_select.simulator import Experiment
from scripts.gpu.experiments.run_r10_online_diagnostic import COST, MIB

BLOCK = 512
KV_PER_TOKEN = 32768
RECURRENT = 52690944
CONTEXT = 3072
WINDOW = 3  # physical bring-up only; rename and freeze before timing


def digest(path):
    with path.open('rb') as f:
        return file_digest(f, 'sha256').hexdigest()


def prefix_digest(tokens):
    return sha256(pack(f'<{len(tokens)}I', *tokens)).hexdigest()


def rows(path, compressed=False):
    op = gzip.open if compressed else open
    with op(path, 'rt', encoding='utf-8') as f:
        return [json.loads(row) for row in f]


def load_bailian(manifest, directory):
    m = json.loads(manifest.read_bytes())
    for case in m['cases']:
        p = directory / case['generated_filename']
        assert digest(p) == case['generated_sha256']
        all_rows = rows(p)
        assert len(all_rows) == case['requests']
        requests = [dict(id=str(r['request']), tokens=r['tokens'], arrival_ms=r['arrival_ms'],
                         boundaries=[0, *(c['depth'] for c in r['checkpoints'])]) for r in all_rows]
        yield dict(id=p.stem, family='bailian', split=case['split'], requests=requests, source_sha256=digest(p))


def load_oasst(manifest, private):
    m = json.loads(manifest.read_bytes())
    assert digest(private) == m['private_tokens_sha256']
    tokens = iter(rows(private, compressed=True))
    for block in m['blocks']:
        b = []
        for case in block['requests']:
            row = next(tokens)
            assert row['block'] == block['id'] and row['request'] == case['request']
            assert prefix_digest(row['tokens']) == case['sha256']
            b.append(dict(id=case['request'], tokens=row['tokens'], arrival_ms=str(case['tick']),
                          boundaries=case['eligible_boundaries']))
        yield dict(id='oasst-' + block['id'], split=block['split'], family='oasst', requests=b,
                   source_sha256=digest(private))
    assert next(tokens, None) is None


def make_plan(source):
    objects, index, planned = [], {}, []
    sharing = Counter()

    def object_id(kind, tokens, start, end):
        prefix = tuple(tokens[:end])
        key = (kind, prefix_digest(prefix))
        if key not in index:
            i = index[key] = len(objects)
            if kind == 'K':
                spec = dict(kind='attention_kv', start_token_exclusive=start, end_token_inclusive=end,
                            bytes=(end - start) * KV_PER_TOKEN, tier='RAM')
            else:
                spec = dict(kind='recurrent_and_conv_checkpoint', boundary_tokens=end, bytes=RECURRENT, tier='SSD')
            objects.append(dict(id=str(i), prefix_token_runs=[dict(token_id=t, count=1) for t in prefix], **spec))
        i = index[key]
        sharing[i] += 1
        assert tuple(r['token_id'] for r in objects[i]['prefix_token_runs']) == prefix
        return str(i)

    assert source['requests'] and source['family'] in ('bailian', 'oasst', 'synthetic')
    for r in source['requests']:
        t = r['tokens']
        assert 0 < len(t) <= 2560 and min(t) >= 0 and max(t) < 250000
        boundaries = sorted(set(r['boundaries']))
        assert boundaries[0] == 0 and not any(c % BLOCK or c > len(t) for c in boundaries)
        candidates = [dict(boundary_tokens=0, load_order=[])]
        for c in boundaries[1:]:
            keys = [object_id('K', t, e - BLOCK, e) for e in range(BLOCK, c + 1, BLOCK)]
            keys.append(object_id('R', t, c, c))
            candidates.append(dict(boundary_tokens=c, load_order=keys))
        planned.append(dict(id=r['id'], arrival_ms=r['arrival_ms'], input_token_runs=[dict(token_id=i, count=1) for i in t],
                            candidates=candidates))
    assert [Fraction(r['arrival_ms']) for r in planned] == sorted(Fraction(r['arrival_ms']) for r in planned)
    trace = dict(units=dict(bytes_per_mib=MIB), cost=COST, objects=objects,
                 memory=dict(kv_block_tokens=BLOCK, extra_alignment_bytes=0, workspace_mib=0,
                             fixed_mib=5000, execution_mib=256, device_mib=8188),
                 scheduling=dict(future_window_requests=WINDOW, initial_device_object_ids=[], prefetch_release='start'),
                 requests=planned)
    model = dict(context_tokens=CONTEXT, payload=dict(attention_bytes_per_token=KV_PER_TOKEN, checkpoint_bytes=RECURRENT))
    Experiment(trace, model, 140)
    K = sum(x['tier'] == 'RAM' for x in objects)
    summary = dict(id=source['id'], family=source['family'], split=source['split'], source_sha256=source['source_sha256'],
                   requests=len(planned), unique_kv_objects=K, unique_checkpoints=len(objects) - K,
                   referenced_from_multiple_choices=sum(n > 1 for n in sharing.values()),
                   candidate_depths=dict(Counter(c['boundary_tokens'] for r in planned for c in r['candidates'])),
                   payload_bytes=sum(o['bytes'] for o in objects))
    return dict(trace=trace, model=model), summary


def write_plan(source, out):
    result, summary = make_plan(source)
    trace = result['trace']
    prefix = out / source['id']
    p = prefix.with_suffix('.plan')
    with p.open('x', encoding='ascii', newline='\n') as f:
        print('R10CAT1', BLOCK, CONTEXT, len(trace['objects']), len(trace['requests']), file=f)
        for o in trace['objects']:
            print('K' if o['tier'] == 'RAM' else 'R', o['bytes'], file=f)
        for r in trace['requests']:
            tokens = [x['token_id'] for x in r['input_token_runs']]
            print(len(tokens), len(r['candidates']), *tokens, file=f)
            for c in r['candidates']:
                print(c['boundary_tokens'], len(c['load_order']), *c['load_order'], file=f)
    meta = prefix.with_suffix('.schedule.json')
    with meta.open('x', encoding='utf8', newline='\n') as f:
        json.dump(result, f, sort_keys=True, separators=(',', ':'))
        f.write('\n')
    return dict(summary, wire_sha256=digest(p), schedule_sha256=digest(meta))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bailian-manifest', 'bailian-root', 'oasst-manifest', 'oasst-tokens', 'out', 'summary'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    blocks = [*load_bailian(args.bailian_manifest, args.bailian_root), *load_oasst(args.oasst_manifest, args.oasst_tokens)]
    summaries = [write_plan(b, args.out) for b in blocks]
    assert len(summaries) == 22 == len({r['id'] for r in summaries})
    with args.summary.open('x', encoding='utf-8', newline='\n') as f:
        print(json.dumps(dict(scope='exact objects and candidates; clocks and cost table still diagnostic',
                              context_tokens=CONTEXT, kv_block_tokens=BLOCK, blocks=summaries),
                         sort_keys=True, indent=2), file=f)


if __name__ == '__main__':
    main()
