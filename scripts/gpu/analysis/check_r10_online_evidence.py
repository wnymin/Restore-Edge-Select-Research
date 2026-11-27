"""Rebuild chooser snapshots from archived native events without CUDA/Vulkan."""

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
from itertools import groupby
import json
from pathlib import Path

from restore_edge_select.live_snapshot import choose_observed
from scripts.gpu.experiments.run_r10_online_diagnostic import COST, POLICIES, msjson, observation, verify_case


def check(prefix: Path):
    raw = prefix.with_suffix('.csv')
    saved = prefix.with_suffix('.jsonl')
    summary = json.loads(prefix.with_suffix('.json').read_text(encoding='utf-8'))
    assert summary['csv_sha256'] == sha256(raw.read_bytes()).hexdigest()
    assert summary['snapshot_sha256'] == sha256(saved.read_bytes()).hexdigest()
    decisions = [json.loads(line) for line in saved.read_text(encoding='utf-8').splitlines()]
    replayed = 0
    with raw.open(newline='', encoding='utf-8') as f:
        for ((rep, arm), group), row in zip(groupby(csv.DictReader(f), lambda r: (int(r['rep']), r['arm'])),
                                            decisions, strict=True):
            events = [{k: v if k in ('arm', 'event') else int(v) for k, v in r.items()} for r in group]
            decision = next(e for e in events if e['event'] == 'online-decision')
            experiment, snap = observation(events[:decision['event_no'] + 1])
            boundaries = {p: choose_observed(experiment, p, snap) for p in POLICIES}
            assert (rep, arm, boundaries) == (row['rep'], row['arm'], row['replay'])
            assert row['snapshot'] == json.loads(json.dumps(asdict(snap), default=msjson))
            assert row['costs'] == COST and row['capacity_bytes'] == decision['capacity_bytes']
            candidate = int(arm.rsplit('-', 1)[-1]) if row['mode'] == 'forced' else boundaries[arm.rsplit('-', 1)[0]]
            assert candidate == row['chosen']
            verify_case(events, candidate)
            target = [e['event'] for e in events if e['request'] == 1]
            if not candidate:
                assert 'reserve' not in target and 'device-copy' not in target
            elif arm.endswith('-tight') and candidate == 1024:
                assert target.index('reject') < target.index('reserve') < target.index('head-demand') < target.index('device-copy')
            else:
                assert target.index('reserve') < target.index('device-copy') and 'reject' not in target
            replayed += rep >= 0
    assert replayed == summary['arms']
    print(f'{prefix.name}: replayed {replayed} native action cases')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', nargs='+', type=Path)
    for prefix in parser.parse_args().prefix:
        check(prefix)


if __name__ == '__main__':
    main()
