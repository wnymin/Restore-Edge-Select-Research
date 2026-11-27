"""Verify the real-executor snapshot adapter against the frozen R07 replay."""

import argparse
import ast
from collections import Counter
from dataclasses import replace
import gzip
from hashlib import sha256
import inspect
import json
from pathlib import Path
import textwrap

from scripts.cpu.replay.study_r07_arrivals import CONFIG, MODEL, cases, read, run
from restore_edge_select.live_snapshot import audit_simulated_decisions, choose_observed
from restore_edge_select.simulator import Experiment

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "docs/evidence/r08-frozen-simulator.py.gz"


def choose_hash(run_node):
    node = next(x for x in run_node.body if isinstance(x, ast.FunctionDef) and x.name == "choose")
    return sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def audit():
    archived = gzip.decompress(FROZEN.read_bytes())
    old = next(x for x in ast.walk(ast.parse(archived)) if isinstance(x, ast.FunctionDef) and x.name == "_run")
    new = ast.parse(textwrap.dedent(inspect.getsource(Experiment._run))).body[0]
    assert choose_hash(old) == choose_hash(new)
    totals = Counter()
    rules = ('local-cost', 'entry-replay')
    stages = Counter()
    examples = {}
    model = read(MODEL)['models'][0]
    for key, exp in cases(read(CONFIG), 'short'):
        totals['workloads'] += 1
        hidden_future = {}
        for p in rules:
            def audit_one(s, action):
                selected = choose_observed(exp, p, s)
                assert action == selected, (key, p, s, action, selected)
                totals[p] += 1
                stages['active_compute'] += s.compute_start_ms is not None
                stages['active_copy'] += s.compute_start_ms is not None and s.suffix_start_ms is None
                stages['transfer'] += s.transfer_object is not None
                stages['reservations'] += bool(s.admitted)
                stages['same_time_as_latest_arrival'] += s.now_ms == exp.arrivals[s.arrived - 1]
                if s.arrived < len(exp.requests):
                    if s.arrived not in hidden_future:
                        reqs = [dict(r, arrival_ms=str(exp.arrivals[i] + (437 if i >= s.arrived else 0)))
                                for i, r in enumerate(exp.requests)]
                        mib = exp.trace['units']['bytes_per_mib']
                        hidden_future[s.arrived] = Experiment(dict(exp.trace, requests=reqs), model, exp.budget / mib)
                    assert choose_observed(hidden_future[s.arrived], p, s) == action, (key, p, s)
                    totals['changed_unseen_future'] += 1
                if p == 'entry-replay':
                    local = choose_observed(exp, 'local-cost', s)
                    totals['same_snapshot_policy_disagreement'] += local != action
                    if local != action and 'disagreement' not in examples:
                        examples['disagreement'] = {"snapshot_ms": str(s.now_ms), "head": s.head,
                            "request": s.request, "arrived": s.arrived, "entry": action, "local": local, "case": key}
                if s.transfer_object and 'malformed_clock' not in examples:
                    try:
                        choose_observed(exp, p, replace(s, transfer_start_ms=s.now_ms + 1))
                    except ValueError:
                        examples['malformed_clock'] = 'rejected'
                    else:
                        raise AssertionError('Accepted an unobserved future transfer start')
            inspected = audit_simulated_decisions(exp, p, audit_one)
            original = run(exp, p)
            assert all(inspected[field] == original[field] for field in ('boundaries_tokens', 'completion_ms', 'response_sum_ms')), (key, p)
    assert totals['local-cost'] == totals['entry-replay'] == 420 and totals['workloads'] == 60, totals
    assert totals['changed_unseen_future'] and totals['same_snapshot_policy_disagreement'] and all(stages.values()), (totals, stages)
    assert set(examples) == {'disagreement', 'malformed_clock'}
    return {'counts': dict(totals), 'state_coverage': dict(stages), 'examples': examples,
            'frozen_source_sha256': sha256(archived).hexdigest(), 'chooser_ast_sha256': choose_hash(new),
            'scope': 'R07 deterministic stationary simulator -> R10 live snapshot adapter; no GPU action claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    answer = json.dumps(audit(), indent=2, sort_keys=True) + '\n'
    if args.out:
        with args.out.open('x', encoding='utf-8', newline='\n') as f:
            f.write(answer)
    print(answer, end='')


if __name__ == '__main__':
    main()
