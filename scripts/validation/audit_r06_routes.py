"""Count original R05 early returns on zero-overhead R06 holdout replays."""

from collections import Counter, defaultdict
from inspect import getsourcefile
from pathlib import Path
import sys
from types import CodeType

from scripts.cpu.replay.study_r06_transfer import CONFIG, Experiment, ROOT, cases, pack, read, run
from scripts.validation.check_r06_delivery import records


def audit_routes(cfg, frozen, holdout):
    code = next(c for c in Experiment._run.__code__.co_consts if isinstance(c, CodeType) and c.co_name == 'choose')
    source = Path(getsourcefile(code)).read_text(encoding='utf-8').splitlines()
    start = next(i for i, line in enumerate(source) if 'if early_portfolio and not admitted' in line)
    early_line = next(i + 1 for i in range(start, len(source)) if source[i].strip() == 'return base')
    tool = next(i for i in range(6) if sys.monitoring.get_tool(i) is None)
    monitor = sys.monitoring
    monitor.use_tool_id(tool, 'r06-route-audit')
    counts = defaultdict(Counter)
    hits = 0

    def on_line(_, line):
        nonlocal hits
        hits += line == early_line

    monitor.register_callback(tool, monitor.events.LINE, on_line)
    try:
        for (key, exp), recorded in zip(cases(cfg, 'holdout'), holdout, strict=True):
            assert key == recorded['key']
            eligible = sum(len(choices) > 1 for choices in exp.eligible)
            for policy in ('portfolio-v4', 'recalibrated'):
                hits = 0
                monitor.set_local_events(tool, code, monitor.events.LINE)
                result = run(exp, key, policy, frozen)
                monitor.set_local_events(tool, code, 0)
                assert pack(exp, result, result)['sequence'] == recorded['zero'][policy]['sequence']
                group = counts[key['model'], key['domain'], policy]
                group['multi_choice'] += eligible
                group['gate_fast'] += hits
            group = counts[key['model'], key['domain'], 'unified']
            group['multi_choice'] += eligible
            group['gate_fast'] += recorded['routes'].get('fast', 0)
            group['callback_fallback'] += recorded['routes'].get('fallback', 0)
    finally:
        monitor.set_local_events(tool, code, 0)
        monitor.register_callback(tool, monitor.events.LINE, None)
        monitor.free_tool_id(tool)
    return [dict(model=model, cost=cost, policy=p, counts=dict(c),
                 baseline_path_rate=1 - c['gate_fast'] / c['multi_choice'])
            for (model, cost, p), c in sorted(counts.items())]


if __name__ == '__main__':
    path = ROOT / 'docs/evidence'
    result = audit_routes(read(CONFIG), read(path / 'r06-frozen.json'), records(path / 'r06-holdout.jsonl.gz'))
    assert result == read(path / 'r06-routes.json'), 'R06 route audit differs from archive'
    print(f'R06 route audit checked {len(result)} strata.')
