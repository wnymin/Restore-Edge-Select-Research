"""R08-CPU-03: reproducible same-input interventions on archived cases only."""
import argparse
from collections import defaultdict
from copy import copy
from fractions import Fraction
import gzip
import inspect
from itertools import product
import json
from math import prod
from pathlib import Path
import sys

from scripts.cpu.replay import study_r07_arrivals as arrivals
from scripts.cpu.replay import study_r07_extension as extension
from scripts.cpu.replay import study_r08_capacity as capacity
from scripts.cpu.replay.study_r07_diagnostics import capture, diagnose, future_pair
from restore_edge_select.offline_bounds import search

EVIDENCE = capacity.ROOT / 'docs/evidence'
B = 'qwen_traceB_blksz_16.jsonl'
R08_KEYS = {(481, '27b', 16, '512'), (481, '4b', 12, '176'), (482, '27b', 16, '384')}


def records(name):
    with gzip.open(EVIDENCE / f'{name}.jsonl.gz', 'rt', encoding='utf8') as source:
        return {tuple(row['key'].items()): row for row in map(json.loads, source)}


def preflight(exp, record):
    frozen = arrivals.run(exp, 'entry-replay')
    assert arrivals.sequence(frozen) == record['zero']['entry-replay']['sequence'], record['key']
    return frozen


def fixed_prefix(exp, plan):
    # Truncated executor: a different experiment, not the full-window objective.
    return exp._run('fixed', 0, plan, 0, len(plan))


def tail(exp, record, frozen, model, affected=(15, 16)):
    local = arrivals.run(exp, 'local-cost')
    assert arrivals.sequence(local) == record['zero']['local-cost']['sequence']
    local_response = arrivals.response(exp, local)
    touched = [i for i, (a, b) in enumerate(zip(frozen['boundaries_tokens'], local['boundaries_tokens'], strict=True)) if a != b]
    assert touched[0] == 17 and all(frozen['boundaries_tokens'][i] == local['boundaries_tokens'][i] for i in affected)

    assert exp.release_at == 'start'

    def breakdown(result, i):
        a = exp.arrivals[i]
        end = result['completion_ms'][i]
        head = a + result['queue_wait_ms'][i]
        start = head + result['head_to_compute_ms'][i]
        rid = exp.requests[i]['id']
        copy = next((t for t in result['transfers'] if t['request'] == rid and t['mode'] == 'handoff'), None)
        dispatch = copy['start_ms'] if copy is not None else start
        reserved = result['prefetch_reserved_at_ms'][i]
        return dict(index=i, id=rid, boundary=result['boundaries_tokens'][i],
                    arrival=a, head=head, dispatch=dispatch, compute=start, completion=end,
                    reserved=reserved, reservation_release=dispatch if reserved is not None else None,
                    device_copy=copy, response=end-a)

    def detail(result):
        # Include even transfers issued for more distant requests if they overlap
        # the causal interval; timeline entries supply both issue and ready times.
        start = min(exp.arrivals[i] for i in affected)
        stop = max(result['completion_ms'][i] for i in affected)
        return dict(mean=result['response_mean_ms'], requests=[breakdown(result, i) for i in range(14, 19)],
                    transfers=[t for t in result['transfers'] if t['start_ms'] <= stop and t['end_ms'] >= start])

    def summarize(index, action, result):
        rr = arrivals.response(exp, result)
        return dict(index=index, action=action, mean_extra=result['response_mean_ms']-frozen['response_mean_ms'],
                    extra=[rr[i]-local_response[i] for i in affected], worst=max(a-b for a,b in zip(rr, local_response)))

    interventions = []
    reverted = None
    for i in touched:
        actions = exp.eligible[i] if i == touched[0] else (local['boundaries_tokens'][i],)
        for c in actions:
            result = arrivals.run(exp, 'entry-replay', overrides={i:c})
            interventions.append(summarize(i,c,result))
            if i == 17 and c == local['boundaries_tokens'][i]:
                reverted = result
    assert reverted is not None
    assert summarize(17, local['boundaries_tokens'][17], reverted)['extra'] == [0, 0]

    # Also intervene in the reverse direction, keeping the rest local-cost.
    c = frozen['boundaries_tokens'][17]
    reverse = arrivals.run(exp, 'local-cost', overrides={17:c})
    assert [r-local_response[i] for i,r in zip(affected, (arrivals.response(exp, reverse)[i] for i in affected))] == [
        arrivals.response(exp, frozen)[i]-local_response[i] for i in affected]

    # Mechanism ablation only: hold all actions fixed and remove overlap slowdown.
    no_slowdown = capacity.Experiment(dict(exp.trace, overlap_slowdown={}), model, Fraction(exp.budget, capacity.MIB))
    off = [no_slowdown.run('fixed', boundaries=z['boundaries_tokens']) for z in (frozen,reverted)]
    overlap_ablation = [off[0]['completion_ms'][i]-off[1]['completion_ms'][i] for i in affected]
    assert overlap_ablation == [0, 0]

    # Exhaustive Pareto front for the first 20 requests, including every legal
    # boundary there. Fixed prefix simulations share the same initial state.
    n = 20
    local_prefix = fixed_prefix(exp, local['boundaries_tokens'][:n])
    by_tail = {}
    count = 0
    for plan in product(*exp.eligible[:n]):
        count += 1
        z = fixed_prefix(exp, plan)
        extra = max(0, *(a-b for a,b in zip(z['completion_ms'], local_prefix['completion_ms'], strict=True)))
        pair = (z['response_mean_ms'], plan)
        if extra not in by_tail or pair < by_tail[extra]:
            by_tail[extra] = pair
    best = None
    front = []
    for extra, (mean, plan) in sorted(by_tail.items()):
        if best is None or mean < best:
            front.append(dict(worst_extra=extra, mean=mean, plan=plan))
            best = mean
    return dict(key=record['key'], first_plan_difference=touched[0], affected=list(affected),
                frozen=detail(frozen), local=detail(local), reverted=detail(reverted),
                reverse=summarize(17,c,reverse), reverse_events=detail(reverse), interventions=interventions,
                overlap_slowdown_removed_completion_delta=overlap_ablation,
                frontier=dict(n=n, combinations=count, points=front))


def constrained_bounds(exp, target, incumbent_plans, prefix, node_budget):
    """Offline certificates per first action, conditional on the committed prefix."""
    n = len(exp.arrivals)
    out = {}
    for c in exp.eligible[target]:
        restricted = copy(exp)
        first = [*prefix,c]
        restricted.eligible = [*( (a,) for a in first), *exp.eligible[target+1:]]
        seeds = [restricted.run('fixed',boundaries=(*first,*p[target+1:])) for p in incumbent_plans]
        lo,hi,plan,nodes = search(restricted,seeds,node_budget,resource_floor=True,dominance=True)
        assert lo <= hi == restricted.run('fixed',boundaries=plan)['response_sum_ms']
        out[c] = dict(lower=lo/n, upper=hi/n, visited=nodes, plan=plan)
    return out


def capture_isolation(exp, target):
    source,start = inspect.getsourcelines(capacity.Experiment._run)
    line = next(start+j for j,row in enumerate(source) if row.strip() == 'return serial_choice(i)')
    code = capacity.Experiment._run.__code__
    filename = code.co_filename
    data = {}

    def local(frame,event,_):
        if event == 'line' and frame.f_lineno == line:
            v = frame.f_locals
            if v['i'] == target:
                assert v['head'] == target and v['arrived'] == target+1
                data.update(state=(v['now'],v['head'],v['arrived'],v['running_until'],v['transfer'],
                                  tuple(sorted(v['ready'])),tuple(sorted(v['admitted']))),
                            prefix=tuple(v['selected'][:target]), choices=tuple(exp.eligible[target]))
        return local

    def trace(frame,event,_):
        if event == 'call' and frame.f_code.co_filename == filename and frame.f_code.co_name == 'choose':
            return local
        return None

    prior = sys.gettrace()
    sys.settrace(trace)
    try:
        result = arrivals.run(exp,'entry-replay')
    finally:
        sys.settrace(prior)
    return data,result


def bounded_future_audit(exp, state, record, target, model, node_budget=384, observer=capture):
    plan = record['plan']
    seen = state['state'][2]
    if seen == len(exp.arrivals):
        return dict(status='all-arrived')
    now = state['state'][0]
    # Compare one finite, prespecified extension of the arrival spacing.
    delay = Fraction(200)
    requests = [dict(r,arrival_ms=str(exp.arrivals[j] if j < seen else
                    now+delay+exp.arrivals[j]-exp.arrivals[seen])) for j,r in enumerate(exp.requests)]
    other = capacity.Experiment(dict(exp.trace, requests=requests), model, Fraction(exp.budget,capacity.MIB))
    counterpart, _ = observer(other,target)
    if counterpart != state:
        return dict(status='snapshot-changed')
    incumbent_plans = (plan, record['oracle']['plan'])
    before = constrained_bounds(exp,target,incumbent_plans,plan[:target],node_budget)
    after = constrained_bounds(other,target,incumbent_plans,plan[:target],node_budget)

    def possible(table):
        min_upper = min(r['upper'] for r in table.values())
        return [c for c,r in table.items() if r['lower'] <= min_upper]

    a,b = possible(before),possible(after)
    return dict(status='conflict-certified' if not set(a)&set(b) else 'conflict-not-established',
                same_snapshot=True, node_budget=node_budget, future_delay_ms=delay,
                original=before, alternate=after, possible_original=a, possible_alternate=b)


def classify(exp, record, *, model=None, investigate_future=False):
    frozen = preflight(exp, record)
    result = diagnose(exp, record)
    assert result is not None
    if not isinstance(result, tuple):
        i = result['target']
        # Same prefix and through-target truncation: diagnose isolation separately,
        # without presenting a future oracle action as an online feature.
        closed = {}
        prefix = frozen['boundaries_tokens'][:i]
        for c in exp.eligible[i]:
            closed[c] = dict(through_target=fixed_prefix(exp, (*prefix,c))['response_mean_ms'],
                             full=arrivals.run(exp,'entry-replay',overrides={i:c})['response_mean_ms'])
        isolated, _ = capture_isolation(exp,i)
        if isolated and model is not None:
            result['future_audit'] = bounded_future_audit(exp,isolated,record,i,model,observer=capture_isolation)
        result.update(actions=closed, current=frozen['boundaries_tokens'][i], oracle_action=record['oracle']['plan'][i])
        return result
    diag, state = result
    i = diag['target']
    a = frozen['response_mean_ms']
    best = Fraction(str(record['oracle'].get('upper_bound', record['oracle']['upper'])))
    diag['offline'] = dict(upper=best, exact=record['oracle']['exact'], gap_to_feasible=a-best)
    diag['prefix_differences_from_oracle'] = [j for j,(x,y) in enumerate(zip(record['plan'],record['oracle']['plan'],strict=True)) if x!=y]
    diag['interventions'] = {}
    for tag, boundary, tails in (
            ('same-horizon-no-coarse-no-pruning', diag['window4_choice'], ()),
            ('all-arrived-fixed-followers', diag['all_seen_choice'], ()),
            ('joint-current-only', diag['joint_choice'], ()),
            ('joint-force-branches', diag['joint_choice'], diag['joint_indices'])):
        overrides = {i:boundary}
        overrides.update({j:diag['joint_tails'][boundary][j-i-1] for j in tails})
        z = arrivals.run(exp,'entry-replay',overrides=overrides)
        diag['interventions'][tag] = dict(actions=overrides, gain=a-z['response_mean_ms'],
                                         gap_to_feasible=z['response_mean_ms']-best)
    if investigate_future:
        # Enumerate only bounded suffixes, then require disjoint exact optimum
        # first actions at an identical online snapshot before claiming ambiguity.
        combinations = prod(map(len, exp.eligible[i:]))
        diag['future_audit'] = dict(combinations=combinations, attempted=combinations <= 256, pair=None)
        if combinations <= 256:
            pair = future_pair(exp, state, i, record['plan'], exp.trace, model, Fraction(exp.budget, capacity.MIB))
            diag['future_audit']['pair'] = pair
        else:
            diag['future_audit']['bounded'] = bounded_future_audit(exp,state,record,i,model)
    if record['key'].get('window') == 482:
        # Continue locating the residual after the first intervention. The
        # offline plan below is used only to choose diagnostic interventions.
        forced = {i:diag['joint_choice']}
        after = arrivals.run(exp,'entry-replay',overrides=forced)
        diag['oracle_guided_residual'] = []
        for _ in range(len(exp.arrivals)):
            j = next((j for j,(got,want) in enumerate(zip(after['boundaries_tokens'],record['oracle']['plan'],strict=True))
                      if got != want), None)
            if j is None:
                break
            assert j not in forced
            residual_case = dict(record, plan=after['boundaries_tokens'],
                                 zero={'entry-replay':{'mean':float(after['response_mean_ms'])}})
            intervention_exp = copy(exp)
            # Preserve the historical diagnostic entry point and the fixed
            # executor; inject earlier decisions only in this fresh run view.
            def run_after_interventions(policy, **kwargs):
                if policy == 'snapshot-screened':
                    kwargs['decision_overrides'] = {**forced, **(kwargs.get('decision_overrides') or {})}
                return exp.run(policy, **kwargs)
            intervention_exp.run = run_after_interventions
            residual = diagnose(intervention_exp, residual_case)
            forced[j] = record['oracle']['plan'][j]
            nxt = arrivals.run(exp,'entry-replay',overrides=forced)
            diag['oracle_guided_residual'].append(dict(index=j, action=forced[j],
                incremental_gain=after['response_mean_ms']-nxt['response_mean_ms'], gap_to_feasible=nxt['response_mean_ms']-best,
                snapshot_diagnostic=residual[0] if isinstance(residual,tuple) else residual))
            after = nxt
    return diag


def tail_control(exp, record, witness, index, decision, action):
    frozen = preflight(exp, record)
    local = arrivals.run(exp, 'local-cost')
    assert arrivals.sequence(local) == record['zero']['local-cost']['sequence']
    baseline = arrivals.response(exp, local)[index]
    intervention = arrivals.run(exp,'entry-replay',overrides={decision:action})
    assert local['boundaries_tokens'][decision] == action
    return dict(key=record['key'], index=index, request_id=exp.requests[index]['id'],
                decision=decision, action=action, source=witness,
                original_extra=arrivals.response(exp,frozen)[index]-baseline,
                forced_extra=arrivals.response(exp,intervention)[index]-baseline,
                mean_cost=intervention['response_mean_ms']-frozen['response_mean_ms'])


def filter_r08(key):
    return key['trace'] == B and key['section'] == 'fixed' and (
        key['window'],key['model'],key['rate'],key['level']) in R08_KEYS


def collect():
    out = defaultdict(list)
    cfg = capacity.read(capacity.CONFIG)
    cfg['holdout'] = {B:[481,482]}
    rows = records('r08-holdout')
    for key,data,model,scale in capacity.cases(cfg,'holdout'):
        if filter_r08(key):
            exp = capacity.open_exp(data,model,key['rate'],key['capacity_bytes'],scale)
            record = rows[tuple(key.items())]
            r = (tail(exp,record,preflight(exp,record),model) if key['level'] == '512' else classify(exp,record))
            out['r08'].append(r)
    assert len(out['r08']) == 3

    rows = records('r07-holdout')
    witnesses = capacity.read(EVIDENCE / 'r07-holdout-witness.json')
    unresolved = [w['key'] for w in witnesses if w['current'] == w['window4_choice'] == w['all_seen_choice'] == w['joint_choice']]
    assert len(unresolved) == 5
    r07_control = capacity.read(EVIDENCE / 'r07-tail-witness.json')
    for key,exp in arrivals.cases(arrivals.read(arrivals.CONFIG),'holdout'):
        if key == r07_control['key']:
            row = r07_control['checks'][0]
            out['tail_controls'].append(tail_control(exp,rows[tuple(key.items())], 'r07-tail-witness.json',
                r07_control['request'],row['j'],row['token_baseline']))
        if key in unresolved:
            out['r07_unresolved'].append(classify(exp,rows[tuple(key.items())],
                model=capacity.read(capacity.MODEL)['models'][0],investigate_future=True))
    assert len(out['r07_unresolved']) == 5

    rows = records('r07-extension-holdout')
    summary = capacity.read(EVIDENCE / 'r07-extension-summary.json')
    diagnostic = summary['diagnostics']
    r07_extension_control = next(w for w in summary['tail_causal']['witnesses']
        if w['key']['window'] == 471 and w['key']['model'] == '27b')
    keys = diagnostic['screen'] + diagnostic['isolated-path/unresolved']
    for key,data,model,scale,_ in extension.cases(capacity.read(extension.CONFIG),'holdout'):
        if key == r07_extension_control['key']:
            exp = extension.experiment(data,key,model,scale)
            w = r07_extension_control
            out['tail_controls'].append(tail_control(exp,rows[tuple(key.items())], 'r07-extension-summary.json',
                w['request'],w['decision'],w['boundary'][1]))
        if key in keys:
            exp = extension.experiment(data,key,model,scale)
            out['r07_extension'].append(classify(exp,rows[tuple(key.items())],model=model))
    assert len(out['r07_extension']) == len(keys)
    assert len(out['tail_controls']) == 2
    out['input_sha256'] = {name:capacity.file_hash(EVIDENCE / name) for name in (
        'r08-holdout.jsonl.gz','r07-holdout.jsonl.gz','r07-holdout-witness.json','r07-tail-witness.json',
        'r07-extension-holdout.jsonl.gz','r07-extension-summary.json')}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = collect()
    with args.out.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2, default=lambda x: str(x) if isinstance(x,Fraction) else sorted(x))
        stream.write('\n')


if __name__ == '__main__':
    main()
