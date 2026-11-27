"""Build the R10 pilot only cost table; never authorize a held-out run."""

import argparse
from fractions import Fraction
import json
from pathlib import Path

from scripts.validation.check_r10_native_catalog import digest


def load(path):
    return json.loads(path.read_bytes(), parse_float=Fraction)


def anchored(n, ms):
    return dict(startup_ms=0, bytes_per_ms=str(n / ms))


def build(recompute, transport, receipt):
    p, t, run = map(load, (recompute, transport, receipt))
    assert p['receipt_sha256'] == digest(receipt) and run['return_code'] == 0
    assert t['records'] >= 48 and p['samples'] >= 432
    row, = (x for x in t['phase_costs'] if x['tier'] == 'SSD' and x['bytes'] == 52690944)
    phases = row['phases']
    costs = dict(recompute=p['provisional_recompute_cost'],
                 ram_to_device={k: str(v) for k, v in t['provisional_ram_fit'].items()},
                 ssd_to_device=anchored(row['bytes'], phases['end_to_end']['median_ms']),
                 device_copy=anchored(row['bytes'], phases['device_scatter_16_spans']['median_ms']))
    return dict(scope='R10 provisional isolated costs for calibration only', model_sha256=run['model_sha256'],
                cost=costs, source_sha256=dict(prefill=digest(recompute), transfer=digest(transport)),
                limits=dict(ssd_and_device_copy='one actual size only; device copy is provisional contiguous scatter',
                            overlap='not folded into chooser; measured separately', authorization='pilot input only'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('prefill', 'transport', 'prefill-receipt', 'out'):
        ap.add_argument('--' + name, required=True, type=Path)
    a = ap.parse_args()
    result = build(a.prefill, a.transport, a.prefill_receipt)
    with a.out.open('x', encoding='utf8', newline='\n') as f:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2), file=f)


if __name__ == '__main__':
    main()
