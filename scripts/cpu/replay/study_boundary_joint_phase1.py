"""Show why pairwise feasibility and gross byte totals are insufficient."""

from copy import deepcopy
from fractions import Fraction
from itertools import product

from scripts.cpu.replay.study_boundary_window_phase1 import Experiment, ROOT, make_asymmetric, make_trace, read


def trace_for(mode, model, *, free_case=False):
    trace = make_trace(2)
    if free_case:
        make_asymmetric(trace, model, block=32, short_boundaries=(896, 1120))
    else:
        make_asymmetric(trace, model)
    trace['scheduling']['future_window_requests'] = 3
    if mode == 'c-only':
        return trace
    peer = deepcopy(trace['requests'][2])
    peer.update(id='D', arrival_ms=3)
    alias = mode == 'alias'
    if not alias:
        peer['input_token_runs'][0]['token_id'] = 40
    copies = {}
    for item in tuple(trace['objects']):
        key = item['id']
        if not key.startswith('C.') or alias and item['kind'] == 'attention_kv':
            continue
        own = deepcopy(item)
        own['id'] = copies[key] = 'D' + key[1:]
        own['prefix_token_runs'][0]['token_id'] = peer['input_token_runs'][0]['token_id']
        trace['objects'].append(own)
    for candidate in peer['candidates']:
        candidate['load_order'] = [copies.get(key, key) for key in candidate['load_order']]
    if mode == 'd-only':
        trace['requests'].pop()
    trace['requests'].append(peer)
    return trace


def main():
    model = read(ROOT / 'configs/qwen3.5-4b.json')['models'][0]
    expected = {('c-only', 244): Fraction(52, 9), ('d-only', 244): Fraction(52, 9),
                ('distinct', 244): Fraction(-17, 9), ('alias', 244): Fraction(20, 3),
                ('distinct', 256): Fraction(46, 9)}
    print('mode cap gain_of_deep deep_union deep_peak D_deep_reservation')
    for (mode, cap), gain in expected.items():
        trace = trace_for(mode, model)
        exp = Experiment(trace, model, cap)
        plan = lambda b: [0, b] + [1152] * (len(trace['requests']) - 2)
        shallow, deep = (exp.run('fixed', boundaries=plan(b)) for b in (768, 1152))
        footprint = exp.size({key for r in trace['requests'][1:] for key in r['candidates'][-1]['load_order']})
        assert shallow['response_sum_ms'] - deep['response_sum_ms'] == gain
        assert shallow['completion_ms'][1] - deep['completion_ms'][1] == Fraction(26, 9)
        assert (mode == 'distinct' and cap == 244) == (footprint > exp.budget)
        reserved = deep['prefetch_reserved_at_ms'][-1]
        assert (reserved is None) == (footprint > exp.budget)
        print(mode, cap, f'{float(gain):+.3f}', footprint / (1 << 20),
              deep['peak_prefetch_bytes'] / (1 << 20), reserved)

    joint = {('c-only', 248): Fraction(316, 81), ('d-only', 248): Fraction(316, 81),
             ('distinct', 248): Fraction(-44, 81), ('alias', 248): Fraction(104, 27),
             ('distinct', 256): Fraction(280, 81)}
    print('free choices: mode cap combinations gain_of_deep unique_optimum')
    for (mode, cap), expected in joint.items():
        exp = Experiment(trace_for(mode, model, free_case=True), model, cap)
        tails = list(product((0, 1120, 1152), repeat=len(exp.requests) - 2))
        by_target = {b: min(exp.run('fixed', boundaries=[0, b, *tail])['response_sum_ms'] for tail in tails)
                     for b in (0, 896, 1152)}
        gain = by_target[896] - by_target[1152]
        oracle = exp.offline()
        assert gain == expected and by_target[0] > max(by_target[896], by_target[1152])
        assert oracle['combinations_evaluated'] == 3 * len(tails)
        best = [0, 896 if gain < 0 else 1152] + [1152] * (len(exp.requests) - 2)
        assert oracle['offline_optima'] == [best]
        print(mode, cap, oracle['combinations_evaluated'], f'{float(gain):+.3f}', best)


if __name__ == '__main__':
    main()
