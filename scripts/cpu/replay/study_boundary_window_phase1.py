"""Compare the analytic window example with the archived Phase 1 fixture."""

from fractions import Fraction
from itertools import product

from scripts.cpu.replay.study_bailian import Experiment, ROOT, read


def make_trace(arrival):
    trace = read(ROOT / 'fixtures/phase1-hand-trace.json')
    trace.pop('expected_by_prefetch_mib')
    trace['requests'][2]['arrival_ms'] = arrival
    path = trace['cost']['ssd_to_device']
    path['bytes_per_ms'] = str(Fraction(path['bytes_per_ms'], 3))
    return trace


def observe_target(exp, arrival):
    seen = {}

    def observe(i, now, depth, costs):
        if i == 1:
            seen['B'] = (now, depth, costs)

    exp.run('remaining-wait-live', observe_candidates=observe)
    assert seen['B'][0] == 1 and seen['B'][0] < arrival
    return seen['B']


def make_asymmetric(trace, model, *, block=128, short_boundaries=(768, 1024)):
    trace['memory']['kv_block_tokens'] = block
    trace['objects'].clear()
    payload = model['payload']
    for request, short in zip(trace['requests'][1:], short_boundaries, strict=True):
        name, run = request['id'], request['input_token_runs'][0]
        length = run['count']
        kvs = []
        for start in range(0, length, block):
            end = min(start + block, length)
            kvs.append(f'{name}.kv{end}')
            trace['objects'].append(dict(id=kvs[-1], kind='attention_kv', tier='RAM',
                start_token_exclusive=start, end_token_inclusive=end,
                prefix_token_runs=[{**run, 'count': end}], bytes=(end - start) * payload['attention_bytes_per_token']))
        for j, end in enumerate((short, length), 1):
            state = f'{name}.state{end}'
            trace['objects'].append(dict(id=state, kind='recurrent_and_conv_checkpoint', tier='SSD',
                boundary_tokens=end, prefix_token_runs=[{**run, 'count': end}], bytes=payload['checkpoint_bytes']))
            request['candidates'][j] = dict(boundary_tokens=end, load_order=kvs[:end // block] + [state])


def main():
    model = read(ROOT / 'configs/qwen3.5-4b.json')['models'][0]
    candidates = (0, 576, 1152)
    snapshots = {}
    expected = {(160, 2): (576, 576, Fraction(-5, 2)),
                (160, 40): (1152, 0, Fraction(17, 4)),
                (192, 2): (1152, 1152, Fraction(17, 2)),
                (192, 40): (1152, 0, Fraction(17, 4))}
    print('cap C_arrival B_best_if_C_fixed B_and_C_best_if_joint B_deep_gain_if_C_fixed')
    for cap, arrival in product((160, 192), (2, 40)):
        exp = Experiment(make_trace(arrival), model, cap)
        snap = observe_target(exp, arrival)
        assert snap == snapshots.setdefault(cap, snap)
        choices = {b: exp.run('fixed', boundaries=[0, b, 1152]) for b in candidates}
        best = min(choices, key=lambda b: choices[b]['response_mean_ms'])
        shallow, deep = choices[576], choices[1152]
        assert shallow['completion_ms'][1] - deep['completion_ms'][1] == Fraction(17, 4)
        gain = 3 * (shallow['response_mean_ms'] - deep['response_mean_ms'])
        joint = exp.offline()['boundaries_tokens']
        assert (best, joint[2], gain) == expected[cap, arrival] and joint[1] == 1152
        print(cap, arrival, best, '/'.join(map(str, joint[1:])), f'{float(gain):+.2f} ms')

    print('Asymmetric candidates: cap C_arrival unique_B/C_optimum B_deep_gain_after_optimizing_C')
    snapshots.clear()
    expected = {(160, 2): ([0, 768, 1152], Fraction(-20, 9)),
                (171, 2): ([0, 1152, 1152], Fraction(52, 9)),
                (160, 40): ([0, 1152, 0], Fraction(26, 9)),
                (171, 40): ([0, 1152, 0], Fraction(26, 9))}
    for cap, arrival in product((160, 171), (2, 40)):
        trace = make_trace(arrival)
        make_asymmetric(trace, model)
        exp = Experiment(trace, model, cap)
        snap = observe_target(exp, arrival)
        assert snap == snapshots.setdefault(cap, snap)
        best_with_b = {b: min(exp.run('fixed', boundaries=[0, b, c])['response_sum_ms']
                              for c in (0, 1024, 1152)) for b in (0, 768, 1152)}
        gain = best_with_b[768] - best_with_b[1152]
        oracle = exp.offline()
        assert oracle['combinations_evaluated'] == 9
        assert oracle['offline_optima'] == [expected[cap, arrival][0]] and gain == expected[cap, arrival][1]
        assert best_with_b[0] > max(best_with_b[768], best_with_b[1152])
        print(cap, arrival, '/'.join(map(str, oracle['boundaries_tokens'][1:])), f'{float(gain):+.2f} ms')


if __name__ == '__main__':
    main()
