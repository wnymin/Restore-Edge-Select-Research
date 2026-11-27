"""Build a provisional per-chunk prefill predictor and score prespecified shapes."""

import argparse
from collections import defaultdict
import csv
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
from statistics import median
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from restore_edge_select.prefill_cost import build_prefill_cost
from scripts.validation.check_r10_native_catalog import digest

FIT = {(p, n) for p in (0, 512, 1024, 1536, 2048, 2560) for n in (1, 16, 64, 128, 256)}
VALIDATION = {(256, 32), (768, 95), (1280, 191), (1792, 240), (2304, 48), (2304, 256)}


def score(path, repeats):
    observed, orders = defaultdict(dict), defaultdict(set)
    with path.open(encoding='ascii', newline='') as f:
        for r in csv.DictReader(f):
            rep, order, pos, n, enqueue, end = map(int, (r[k] for k in ('repeat', 'order', 'position', 'tokens', 'submit_us', 'complete_us')))
            shape = pos, n
            assert 0 <= rep < repeats and 0 <= order < len(FIT | VALIDATION) and shape in FIT | VALIDATION
            assert rep not in observed[shape] and order not in orders[rep]
            assert (shape in FIT) == (r['split'] == 'fit') and 0 <= enqueue <= end and end > 0
            orders[rep].add(order)
            observed[shape][rep] = end
    assert set(observed) == FIT | VALIDATION and all(len(v) == repeats for v in observed.values())
    config = dict(kind='chunk-sampled-prefill', chunk_tokens=256, samples=[
        dict(cached_tokens=p, suffix_tokens=n, ms=[str(Fraction(us, 1000)) for _, us in sorted(observed[p, n].items())])
        for p, n in sorted(FIT)
    ])
    estimate = build_prefill_cost(config, {})
    scores = []
    for p, n in sorted(VALIDATION):
        expected = estimate(p, n)
        measured = median(Fraction(us, 1000) for us in observed[p, n].values())
        scores.append(dict(position=p, tokens=n, predicted_ms=float(expected), median_measured_ms=float(measured),
                           absolute_error_ms=float(abs(expected - measured)),
                           absolute_relative_error=float(abs(expected - measured) / measured)))
    bad, ties, comparable = 0, 0, 0
    for a, b in combinations(scores, 2):
        actual = a['median_measured_ms'] - b['median_measured_ms']
        modeled = a['predicted_ms'] - b['predicted_ms']
        if not actual:
            continue
        comparable += 1
        bad += actual * modeled < 0
        ties += modeled == 0
    return dict(scope='isolated native prefill chunks; no end-to-end or overlap claim', source_sha256=digest(path),
                samples=len(observed) * repeats, fit_shapes=len(FIT), validation_shapes=scores,
                maximum_validation_error_ms=max(x['absolute_error_ms'] for x in scores),
                maximum_validation_relative_error=max(x['absolute_relative_error'] for x in scores),
                rank_reversals=bad, modeled_ties=ties, comparable_pairs=comparable, provisional_recompute_cost=config)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ('input', 'stderr', 'receipt', 'archive', 'out'):
        ap.add_argument('--' + k, required=True, type=Path)
    ap.add_argument('--repeats', type=int, default=12)
    a = ap.parse_args()
    if a.archive.exists() or a.out.exists():
        raise FileExistsError('choose new evidence paths')
    receipt = json.loads(a.receipt.read_bytes())
    assert receipt['return_code'] == 0 and a.repeats == receipt['repeats']
    evidence = score(a.input, a.repeats)
    with ZipFile(a.archive, 'x') as z:
        for path in (a.input, a.stderr, a.receipt):
            z.writestr(ZipInfo(path.name), path.read_bytes(), compress_type=ZIP_DEFLATED)
    evidence['archive_sha256'] = digest(a.archive)
    evidence['receipt_sha256'] = digest(a.receipt)
    with a.out.open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2), file=f)
    print(f"Measured {evidence['samples']} prefill chunks; max validation error {evidence['maximum_validation_error_ms']:.2f} ms.")


if __name__ == '__main__':
    main()
