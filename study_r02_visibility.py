"""Separate arrival timing from FIFO changes in the R02 visibility contrast."""
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path

from scripts.cpu.lookahead.staging.study_r02_reference import evaluate, ROOT, read


def study():
    inputs = read(ROOT / 'docs/evidence/r02-cpu-inputs.json')
    variants = {'original': inputs}
    for mode in ('same_fifo', 'file_order'):
        data = deepcopy(inputs)
        for case in data['cases']:
            requests = case['trace']['requests']
            if mode == 'same_fifo':
                requests.sort(key=lambda request: F(request['arrival_ms']))
            for request in requests:
                request['arrival_ms'] = 0
        variants[mode] = data

    result = {}
    for split in ('design', 'holdout'):
        tables = {mode: evaluate(data, split) for mode, data in variants.items()}
        rows = []
        for original, same_fifo, file_order in zip(*tables.values(), strict=True):
            assert original['id'] == same_fifo['id'] == file_order['id']
            assert original['order'] == same_fifo['order']
            rows.append(dict(id=original['id'], original=original, same_fifo=same_fifo, file_order=file_order))
        result[split] = rows
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', type=Path)
    args = parser.parse_args()
    result = study()
    if args.check:
        assert result == read(args.check)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    for split, rows in result.items():
        print(split, len(rows), {
            label: sum(row[a]['winner'] != row[b]['winner'] for row in rows)
            for label, a, b in [('arrival', 'original', 'same_fifo'),
                                ('fifo', 'same_fifo', 'file_order'),
                                ('joint', 'original', 'file_order')]
        })
