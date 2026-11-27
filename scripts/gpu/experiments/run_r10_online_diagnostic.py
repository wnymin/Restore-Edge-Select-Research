"""Exercise the frozen chooser at a native restore event; validate emitted actions."""

import argparse
import csv
from dataclasses import asdict
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
from subprocess import PIPE, Popen
from time import perf_counter_ns

from restore_edge_select.live_snapshot import ObservedDecision, choose_observed
from restore_edge_select.simulator import Experiment

MIB = 1 << 20
IDS = ('kv0', 'r512', 'kv1', 'r1024')
POLICIES = ('entry-replay', 'local-cost')
COST = {  # Rounded preflight estimates, fixed before this action diagnostic.
    'ram_to_device': dict(startup_ms=0, bytes_per_ms=12_000_000),
    'ssd_to_device': dict(startup_ms=0, bytes_per_ms=1_550_000),
    'device_copy': dict(startup_ms=0, bytes_per_ms=96_000_000),
    'recompute': dict(a_ms_per_token_rational=[1, 2], b_ms_per_pair_rational=[0, 1]),
}


def fixture(sizes, capacity, now, *, added_ssd_ms=0):
    # Token values themselves never enter the chooser. Repeat stable identity
    # labels here; the actual native prefixes and splits are checked in C++.
    def tokens(pairs):
        return [dict(token_id=i, count=n) for i, n in pairs]

    shallow, deep = [(2, 512)], [(2, 512), (3, 512)]
    objects = []
    for i, (name, size) in enumerate(zip(IDS, sizes, strict=True)):
        end = 512 if i < 2 else 1024
        obj = dict(id=name, bytes=size, tier='RAM' if i % 2 == 0 else 'SSD',
                   prefix_token_runs=tokens(shallow if i < 2 else deep))
        if i % 2:
            obj.update(kind='recurrent_and_conv_checkpoint', boundary_tokens=end)
        else:
            obj.update(kind='attention_kv', start_token_exclusive=end - 512, end_token_inclusive=end)
        objects.append(obj)

    def request(rid, t, runs, choices):
        return dict(id=rid, arrival_ms=str(t), input_token_runs=tokens(runs),
                    candidates=[dict(boundary_tokens=c, load_order=IDS[:2] if c == 512 else
                                     (IDS[0], IDS[2], IDS[3]) if c == 1024 else []) for c in choices])

    assert sizes[0] == sizes[2] and sizes[1] == sizes[3] and all(s % 65536 == 0 for s in sizes)
    cost = dict(COST, ssd_to_device=dict(COST['ssd_to_device'], startup_ms=added_ssd_ms))
    trace = dict(objects=objects, units=dict(bytes_per_mib=MIB), cost=cost,
                 memory=dict(kv_block_tokens=512, extra_alignment_bytes=0, workspace_mib=0,
                             fixed_mib=0, execution_mib=128, device_mib=8192),
                 scheduling=dict(initial_device_object_ids=[], future_window_requests=3, prefetch_release='start'),
                 requests=[request('predecessor', 0, [(1, 512)], [0]),
                           request('shallow', 0, shallow + [(4, 8)], [0, 512]),
                           request('target', now, deep + [(5, 8)], [0, 512, 1024])])
    spec = dict(context_tokens=2048, payload=dict(checkpoint_bytes=sizes[1], attention_bytes_per_token=sizes[0] // 512))
    return Experiment(trace, spec, Fraction(capacity, MIB))


def observation(rows):
    one = lambda key: next(r for r in rows if r['event'] == key)
    e, xfer, pred = map(one, ('online-decision', 'upload-submit', 'pred-submit'))
    sizes = [r['bytes'] for r in rows if r['event'] == 'catalog-size']
    assert len(sizes) == 4 and e['status'] in (0, 1) and xfer['fence'] == e['fence'] > 0
    assert one('reserve')['request'] == 0 and e['charged_bytes'] == sum(sizes[:2])
    assert pred['event_no'] < xfer['event_no'] < e['event_no'] and not any(r['event'] == 'pred-end' for r in rows)
    ready = frozenset(IDS[r['object']] for r in rows if r['event'] == 'upload-ready')
    assert ('kv0' in ready) == bool(e['status'])
    witnessed = [r for r in rows if r['event'] == 'pred-query']
    if witnessed:
        query, = witnessed
        assert query['status'] in (0, 1) and query['fence'] > 0 and query['event_no'] + 1 == e['event_no']
        assert not e['arm'].endswith('-after') or query['status'] == 1
    else:
        assert not e['arm'].endswith('-after')
    done = bool(witnessed and witnessed[0]['status'])
    now, start, began = (Fraction(r['wall_us'], 1000) for r in (e, xfer, pred))
    run_start = None if done else began
    snap = ObservedDecision(now, int(done), 3, 2, (0, 512), ((1, IDS[:2]),), ready, (), run_start, run_start,
                            None if ready else 'kv0', None if ready else start)
    return fixture(sizes, e['capacity_bytes'], now), snap


def verify_case(rows, submitted):
    def select(event):
        return [r for r in rows if r['event'] == event]

    assert [r['event_no'] for r in rows] == list(range(len(rows)))
    assert select('online-issued')[0]['bytes'] == submitted
    assert all(r['charged_bytes'] <= r['capacity_bytes'] for r in rows)
    assert [r['bytes'] for r in select('prefix-work')] == [1024 - submitted]
    assert [(r['request'], r['bytes'], r['status']) for r in select('consumed')] == [(0, 8, 1), (1, 8, 1)]
    assert [r['bytes'] for r in select('device-copy') if r['request'] == 1] == (
        [] if submitted == 0 else [69_468_160 if submitted == 512 else 86_245_376])
    assert select('case-end')[0]['charged_bytes'] == 0


def msjson(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, frozenset):
        return sorted(value)
    raise TypeError(type(value))


def run(args):
    env = os.environ | dict(PATH=str(args.runtime.resolve()) + os.pathsep + os.environ['PATH'],
                            GGML_VK_VISIBLE_DEVICES='1', GGML_VK_ASYNC_USE_TRANSFER_QUEUE='1',
                            GGML_VK_R10_CONCURRENT_BUFFERS='1')
    for unnecessary in ('VK_INSTANCE_LAYERS', 'VK_ADD_LAYER_PATH'):
        env.pop(unnecessary, None)
    command = [args.worker, args.model, args.runtime, 'Vulkan0', args.scratch, str(args.repeats),
               'online-forced' if args.forced else 'online-observed' if args.observer else 'online']
    if args.observer:
        env.update(VK_INSTANCE_LAYERS='VK_LAYER_R10_timeline_observe', VK_ADD_LAYER_PATH=str(args.observer.resolve().parent))
        command.append(args.observer.resolve())
    mode = 'forced' if args.forced else 'observed' if args.observer else 'chooser'
    diagnostics = []
    if any(args.scratch.parent.glob(args.scratch.name + '.*')):
        raise ValueError('Use a new diagnostic scratch prefix')
    with (args.output.with_suffix('.csv').open('x', newline='', encoding='utf-8') as raw,
          args.output.with_suffix('.log').open('xb') as err,
          args.output.with_suffix('.jsonl').open('x', encoding='utf-8') as snapshots,
          Popen(command, env=env, stdin=PIPE, stdout=PIPE, stderr=err, text=True, bufsize=1, encoding='ascii') as worker):
        class Tee:
            def __iter__(self):
                for line in worker.stdout:
                    raw.write(line)
                    yield line

        rows = []
        for record in csv.DictReader(Tee()):
            r = {k: v if k in ('arm', 'event') else int(v) for k, v in record.items()}
            rows.append(r)
            arm = r['arm']
            if r['event'] == 'online-decision':
                exp, snap = observation(rows)
                decision_count = len(rows)
                policy = arm.rsplit('-', 1)[0]
                started = perf_counter_ns()
                chosen = int(arm.split('-')[-1]) if args.forced else choose_observed(exp, policy, snap)
                ns = perf_counter_ns() - started
                worker.stdin.write(f'{chosen}\n')
                worker.stdin.flush()
            if r['event'] == 'case-end':
                # Only one policy runs on the live path. Rebuild the original
                # snapshot and evaluate both only after native consumption.
                exp, snap = observation(rows[:decision_count])
                choices = {p: choose_observed(exp, p, snap) for p in POLICIES}
                assert args.forced or chosen == choices[policy]
                verify_case(rows, chosen)
                item = dict(mode=mode, rep=r['rep'], arm=arm, chosen=chosen, replay=choices, choose_ns=ns,
                            snapshot=asdict(snap), capacity_bytes=r['capacity_bytes'], costs=COST,
                            exact_logit_positions=16, physical_inflight=snap.transfer_object is not None)
                snapshots.write(json.dumps(item, sort_keys=True, default=msjson) + '\n')
                diagnostics.append(item)
                rows = []
        assert not rows and worker.wait() == 0, 'incomplete native action log'
    formal = [r for r in diagnostics if r['rep'] >= 0]
    assert len(formal) == args.repeats * (3 if args.forced else 6 if args.observer else 4)
    summary = dict(scope='one native target decision; synthetic arrival is triggered by a hardware poll', mode=mode,
                   arms=len(formal), exact_logit_positions=16 * len(formal),
                   physical_inflight=sum(r['physical_inflight'] for r in formal),
                   boundaries={arm: sorted({r['chosen'] for r in formal if r['arm'] == arm})
                               for arm in sorted({r['arm'] for r in formal})},
                   same_snapshot_disagreements=sum(len(set(r['replay'].values())) == 2 for r in formal),
                   csv_sha256=sha256(args.output.with_suffix('.csv').read_bytes()).hexdigest(),
                   snapshot_sha256=sha256(args.output.with_suffix('.jsonl').read_bytes()).hexdigest())
    if args.observer:
        summary['witnessed_compute'] = {state: sum(bool(r['snapshot']['compute_start_ms'] is None) == done for r in formal)
                                        for state, done in (('pending', False), ('ready', True))}
        assert all(summary['witnessed_compute'].values())
    with args.output.with_suffix('.json').open('x', encoding='utf-8') as f:
        json.dump(summary, f, sort_keys=True, indent=2)
        f.write('\n')
    print(json.dumps(summary, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for attr in ('worker', 'runtime', 'model', 'scratch', 'output'):
        parser.add_argument('--' + attr, required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--forced', action='store_true')
    modes.add_argument('--observer', type=Path, help='Path to the independent Vulkan observer DLL and adjacent layer manifest')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    run(args)


if __name__ == '__main__':
    main()
