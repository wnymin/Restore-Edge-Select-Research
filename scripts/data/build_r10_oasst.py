"""Seal local R10 conversation/coding prompts using OASST1 trees and native Qwen IDs."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256, file_digest
from itertools import combinations
import json
import os
from pathlib import Path
import re
import struct
from subprocess import PIPE, Popen
import unicodedata

from scripts.data.make_r08_calibration_gguf import get, read_exact, skip_values, string


SOURCE = dict(dataset='OpenAssistant/oasst1', revision='fdf72ae0827c1cda404aff25b6603abec9e3399b',
              file='2023-04-12_oasst_ready.trees.jsonl.gz', license='Apache-2.0',
              sha256='2a9a8fd343e9b28e04a895a669d3253f82d93e9c174d440199ae19d5fafbdff7')
MODEL_SHA = '00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4'
SALT = 'r10-oasst-native-v1'
LIMIT = 1792
CHECKPOINT = 512
CODING = re.compile(r'\b(?:debug(?:ging)?|programming|python|javascript|typescript|sql|html|bash|github|compile)\b', re.I)
GENERATION = '<|im_start|>assistant\n<think>\n\n</think>\n\n'


def digest(path):
    with path.open('rb') as f:
        return file_digest(f, 'sha256').hexdigest()


def stable(*values):
    return sha256('\0'.join([SALT, *values]).encode()).hexdigest()


def token_digest(ids):
    return sha256(struct.pack(f'<{len(ids)}I', *ids)).hexdigest()


def template(model):
    with model.open('rb') as f:
        assert read_exact(f, 4) == b'GGUF' and get(f, 'I') == 3
        _, fields = get(f, 'Q'), get(f, 'Q')
        for _ in range(fields):
            name, typ = string(f), get(f, 'I')
            if name == 'tokenizer.chat_template':
                assert typ == 8
                return string(f)
            skip_values(f, typ)
    raise ValueError('Model has no chat template')


def render(path):
    # Text-only branch of the fixed Qwen3.5 GGUF template: no system/tools,
    # historical assistant turns, latest user, generation without thinking.
    return ''.join(f"<|im_start|>{m['role']}\n{m['text'].strip()}<|im_end|>\n" for m in path) + GENERATION


def approved(message, role):
    content = (message.get('text') or '').strip()
    return (message['role'] == role and message['lang'] == 'en' and message.get('review_result') is True
            and not message['deleted'] and not message['synthetic'] and content and len(content) <= 16000
            and not any(tag in content.lower() for tag in ('<|', '<think>', '</think>', '<tool_', '</tool_')))


def user_paths(node, path=()):
    role = 'assistant' if len(path) % 2 else 'prompter'
    if not approved(node, role):
        return
    normalized = dict(node, role='user' if role == 'prompter' else role)
    path = path + (normalized,)
    if role == 'prompter':
        yield path
    for child in node['replies']:
        yield from user_paths(child, path)


def shared(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def read_frame(pipe, length):
    data = pipe.read(length)
    if len(data) != length:
        raise RuntimeError('Native tokenizer did not return a complete frame; see its log')
    return data


def scan(source, worker):
    def tokenize(path):
        payload = render(path).encode('utf-8')
        if len(payload) > 1 << 20:
            return ()
        worker.stdin.write(struct.pack('<I', len(payload)) + payload)
        worker.stdin.flush()
        n, = struct.unpack('<I', read_frame(worker.stdout, 4))
        ids = struct.unpack(f'<{n}I', read_frame(worker.stdout, 4 * n))
        return ids if 128 <= n <= LIMIT else ()

    inventory, per_root = Counter(), {}
    seen_tree = set()
    with gzip.open(source, 'rt', encoding='utf-8') as f:
        for line in f:
            tree = json.loads(line)
            ident, root = tree['message_tree_id'], tree['prompt']
            inventory['source_trees'] += 1
            assert ident not in seen_tree
            seen_tree.add(ident)
            if tree['tree_state'] != 'ready_for_export' or not approved(root, 'prompter'):
                continue
            inventory['reviewed_english_roots'] += 1
            paths = list(user_paths(root))
            if not any(len(p) >= 3 for p in paths):
                continue
            inventory['approved_multiturn_roots'] += 1
            normalized_root = ' '.join(unicodedata.normalize('NFKC', root['text']).casefold().split())
            root_hash = stable(normalized_root)
            ids = {p[-1]['message_id']: tokenize(p) for p in paths}
            choices = []
            for a, b in combinations(paths, 2):
                aid, bid = a[-1]['message_id'], b[-1]['message_id']
                x, y = ids[aid], ids[bid]
                if not x or not y or x == y or max(len(a), len(b)) < 3:
                    continue
                lcp = shared(x, y)
                reuse = min(lcp, len(x) - 1, len(y) - 1) // CHECKPOINT * CHECKPOINT
                chron = sorted((a, b), key=lambda p: (p[-1]['created_date'], p[-1]['message_id']))
                order = tuple(p[-1]['message_id'] for p in chron)
                choices.append((-reuse, stable(ident, *order), lcp, order, tuple(len(p) for p in chron)))
            if not choices:
                continue
            score, pair_key, lcp, order, turns = min(choices)
            coding = bool(CODING.search(root['text']) or chr(96) * 3 in root['text'])
            record = dict(tree_id=ident, root_hash=root_hash, coding_tag=coding, shared_tokens=lcp,
                          common_checkpoint=-score, message_ids=order, turn_counts=turns,
                          ids=[ids[k] for k in order], key=stable(ident, pair_key))
            previous = per_root.get(root_hash)
            if previous:
                inventory['normalized_duplicate_roots'] += 1
            if previous is None or record['key'] < previous['key']:
                per_root[root_hash] = record
    counts = Counter((r['common_checkpoint'] > 0, r['coding_tag']) for r in per_root.values())
    inventory.update({f'eligible_reuse_{int(reuse)}_coding_{int(code)}': counts[reuse, code]
                      for reuse in (False, True) for code in (False, True)})
    return inventory, sorted(per_root.values(), key=lambda r: r['key'])


def build_blocks(pairs, include_service_holdout=False):
    remaining = pairs[:]
    used_prompts = set()
    blocks, private = [], []
    block_count = 12 if include_service_holdout else 10
    for block_num in range(block_count):
        group = []

        def take(n, predicate):
            chosen = []
            for r in list(remaining):
                if len(chosen) >= n:
                    break
                if not predicate(r):
                    continue
                ids = {token_digest(s) for s in r['ids']}
                if not used_prompts.isdisjoint(ids):
                    remaining.remove(r)
                    continue
                used_prompts.update(ids)
                chosen.append(r)
                remaining.remove(r)
            if len(chosen) != n:
                raise ValueError(f'Insufficient disjoint eligible pairs in block {block_num}: needed {n}, found {len(chosen)}')
            group.extend(chosen)

        take(8, lambda r: r['common_checkpoint'] >= CHECKPOINT)
        take(max(0, 4 - sum(r['coding_tag'] for r in group)), lambda r: r['coding_tag'])
        take(32 - len(group), lambda _: True)
        group.sort(key=lambda r: stable(str(block_num), r['tree_id']))
        if block_num < 4:
            split, block_id = 'calibration', f'cal-{block_num + 1:02}'
        elif block_num < 10:
            split, block_id = 'holdout', f'hold-{block_num - 3:02}'
        else:
            split, block_id = 'service-holdout', f'service-hold-{block_num - 9:02}'
        output = []
        # Eight trees overlap within each wave. Ticks are relative units;
        # the actual rate is set separately before the GPU holdout.
        for wave in range(0, 32, 8):
            for j in (0, 1):
                for r in group[wave:wave + 8]:
                    tokens = r['ids'][j]
                    candidates = list(range(0, r['common_checkpoint'] + 1, CHECKPOINT)) if j == 1 else [0]
                    key = f'{block_id}-{len(output):02}'
                    row = dict(request=key, source_tree=stable(r['tree_id']), source_message=stable(r['message_ids'][j]),
                               root_hash=r['root_hash'], coding_tag=r['coding_tag'], tick=len(output),
                               tokens=len(tokens), sha256=token_digest(tokens), turns=r['turn_counts'][j],
                               eligible_boundaries=candidates, exact_common_prefix=r['shared_tokens'] if j else 0)
                    private.append(dict(block=block_id, request=key, tokens=tokens))
                    output.append(row)
        blocks.append(dict(id=block_id, split=split, requests=output))
    roots = Counter(r['source_tree'] for b in blocks for r in b['requests'])
    assert len(roots) == block_count * 32 and set(roots.values()) == {2}
    assert len(used_prompts) == block_count * 64 == len(private)
    return blocks, private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'model', 'worker', 'runtime', 'manifest', 'tokens', 'log'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--append-service-holdout', action='store_true', help='extend the existing selection by two disjoint service blocks')
    args = parser.parse_args()
    assert digest(args.source) == SOURCE['sha256'] and digest(args.model) == MODEL_SHA
    qwen_template = template(args.model)
    assert 'enable_thinking' in qwen_template and GENERATION.partition('\n')[0] in qwen_template
    runtime = args.runtime.resolve()
    env = os.environ | {'PATH': str(runtime) + os.pathsep + os.environ['PATH']}
    with (args.log.open('xb') as log, Popen([args.worker, args.model, runtime / 'ggml-cpu.dll'],
                                           stdin=PIPE, stdout=PIPE, stderr=log, env=env) as worker):
        inventory, pairs = scan(args.source, worker)
        worker.stdin.close()
        assert worker.wait(timeout=15) == 0
    print(json.dumps(inventory, sort_keys=True), flush=True)
    blocks, private = build_blocks(pairs, args.append_service_holdout)
    with (args.tokens.open('xb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as g):
        for row in private:
            g.write((json.dumps(row, separators=(',', ':'), sort_keys=True) + '\n').encode())
    proof = dict(source=SOURCE, model_sha256=MODEL_SHA, gguf_template_sha256=sha256(qwen_template.encode()).hexdigest(),
                 tokenizer_sha256=digest(args.worker), private_tokens_sha256=digest(args.tokens),
                 source_inventory=inventory, arrival_clock='synthetic relative ticks, not observed service arrivals',
                 output_token_limit=128, context_tokens=2048, prompt_token_limit=LIMIT,
                 template_scope='unmodified text turns; no system, tools or thinking; whitespace trimmed by template',
                 branch_note='OASST1 messages follow their own tree paths; assistant history is fixed from corpus',
                 selection='hashed order with minimum 8 natural-reuse pairs and 4 coding-tagged pairs in each block',
                 blocks=blocks)
    with args.manifest.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(proof, f, sort_keys=True, separators=(',', ':'))
        f.write('\n')


if __name__ == '__main__':
    main()
