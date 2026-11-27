"""Validate the two-sequence correctness probe and distinguish batches from microbatches."""

import argparse
import csv
from itertools import product
import json
from math import isfinite
from pathlib import Path
import re

from scripts.validation.check_r10_native_catalog import digest


BEGIN = re.compile(r'^probe-mixed-begin saved-b=([01]) b-first=([01])$')
SPLIT_FIELD = re.compile(r'\b(n_tokens|n_seqs_unq)\s*=\s*(\d+)\s*$')


def mixed_microbatches(lines):
    groups = {}
    case = split = None
    for line in lines:
        start = BEGIN.fullmatch(line)
        if start:
            modes = tuple(map(int, start.groups()))
            if case is not None or modes in groups:
                raise ValueError('duplicate or nested mixed submission marker')
            groups[modes] = case = []
        elif line == 'probe-mixed-end':
            if case is None:
                raise ValueError('mixed submission closed without opening')
            case = split = None
        elif case is not None and 'added ubatch to split:' in line:
            split = {}
            case.append(split)
        elif split is not None and (field := SPLIT_FIELD.search(line)):
            split[field[1]] = int(field[2])
    if case is not None or set(groups) != set(product(range(2), repeat=2)):
        raise ValueError('missing mixed submission trace')
    results = []
    for modes, batches in sorted(groups.items()):
        if not batches or any(set(b) != {'n_tokens', 'n_seqs_unq'} or
                              b['n_tokens'] < b['n_seqs_unq'] or b['n_seqs_unq'] not in (1, 2) for b in batches):
            raise ValueError('missing or invalid internal microbatch sizes; run with LLAMA_BATCH_DEBUG=1')
        if sum(b['n_tokens'] for b in batches) != 65:
            raise ValueError('internal microbatches do not account for the complete mixed submission')
        results.append(dict(restored_b=bool(modes[0]), b_prefill_first=bool(modes[1]), microbatches=batches,
                            any_microbatch_contains_both_sequences=any(b['n_seqs_unq'] == 2 for b in batches)))
    return results


def comparison_rows(rows):
    expectations = set(product(range(2), ('restore_vs_identical_mixed', 'mixed_vs_serial'), ('A', 'B')))
    results = []
    for line in rows:
        kind, seq = line['comparison'], line['sequence']
        order, n, exact, greedy = (int(line[k]) for k in ('b_prefill_first', 'logit_positions', 'exact_positions', 'greedy_positions'))
        expectations.remove((order, kind, seq))
        diff = float(line['max_abs_logit_diff'])
        state = int(line['final_state_exact_or_na'])
        if not (n == 9 and 0 <= exact <= greedy <= n and isfinite(diff) and diff >= 0):
            raise ValueError('invalid validation counts in the probe')
        if kind == 'restore_vs_identical_mixed' and (exact != n or diff != 0 or state != 1):
            raise ValueError('restored and recomputed sequences differ under an identical schedule')
        results.append(dict(b_prefill_first=bool(order), comparison=kind, sequence=seq, logit_positions=n,
                            exact_positions=exact, greedy_positions=greedy, max_abs_logit_diff=diff,
                            final_state_exact=None if state == -1 else bool(state)))
    if expectations:
        raise ValueError('incomplete mixed/reference comparison matrix')
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('csv', 'stderr', 'probe', 'out'):
        p.add_argument('--' + key, required=True, type=Path)
    a = p.parse_args()
    with a.csv.open(encoding='utf-8-sig', newline='') as f:
        rows = comparison_rows(csv.DictReader(f))
    batches = mixed_microbatches(a.stderr.read_text(encoding='utf-8-sig', errors='replace').splitlines())
    with a.out.open('x', encoding='utf-8', newline='\n') as f:
        print(json.dumps(dict(scope='synchronous whole-checkpoint restore in a two-sequence context, no policy timing',
                              csv_sha256=digest(a.csv), stderr_sha256=digest(a.stderr), probe_sha256=digest(a.probe),
                              comparisons=rows, mixed_submissions=batches), indent=2, sort_keys=True), file=f)


if __name__ == '__main__':
    main()
