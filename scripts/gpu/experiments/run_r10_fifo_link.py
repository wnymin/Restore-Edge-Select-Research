"""Close the R10 FIFO controller over native events on the short tensor catalogue."""

import argparse
from dataclasses import asdict
from fractions import Fraction
import json
import os
from pathlib import Path
from queue import Empty, SimpleQueue
from subprocess import PIPE, Popen
from threading import Event, Thread
from time import perf_counter_ns, sleep

from restore_edge_select.live_fifo import Launch, LiveFIFO
from restore_edge_select.live_snapshot import choose_observed
from scripts.gpu.experiments.run_r10_online_diagnostic import IDS, POLICIES, fixture, msjson


def relative_ms(epoch):
    return Fraction(perf_counter_ns() - epoch, 1_000_000)


class Wire:
    def __init__(self, worker, raw):
        self.worker, self.raw = worker, raw

    def receive(self, tag):
        for line in self.worker.stdout:
            if line.startswith('!'):
                k, *args = line.split()
                if k != tag:
                    raise RuntimeError(f'Expected {tag}; native worker replied {k}')
                return tuple(map(int, args))
            self.raw.write(line)
        raise RuntimeError(f'Native FIFO exited before responding with {tag}; inspect native log')

    def command(self, *args, tag='!ok'):
        self.worker.stdin.write(' '.join(map(str, args)) + '\n')
        self.worker.stdin.flush()
        return self.receive(tag)


class GPU:
    def __init__(self, wire, epoch, ssd_hold):
        self.wire, self.epoch = wire, epoch
        self.ssd_hold = ssd_hold
        self.fifo = None
        self.read_ticket = None
        self.pending_ssd = None
        self.effects = []

    def record(self, kind, now, **details):
        self.effects.append(dict(kind=kind, scheduler_ms=now, ack_ms=relative_ms(self.epoch), **details))

    def reserve(self, request, deps, active_kv, now):
        if active_kv not in (frozenset(), frozenset({'kv0'})):
            raise ValueError('The native catalogue can alias only its common kv0 object')
        depth = self.fifo.chosen[request]
        self.wire.command('reserve', request, depth, int(bool(active_kv)))
        self.record('reserve', now, request=request, deps=deps, active_kv=active_kv)

    def read(self, action, now):
        assert self.read_ticket is self.pending_ssd is None
        depth = self.fifo.chosen[action.request]
        payload = ('read', action.request, depth, IDS.index(action.object_id), action.lane_use)
        self.record('read', now, action=asdict(action))
        effect = self.effects[-1]
        if self.ssd_hold and action.object_id.startswith('r'):
            # Deliberate software hold before an actual unbuffered SSD read.
            # Do not block the client or compute observation during this hold.
            effect['hold_earliest_ms'] = effect['ack_ms'] + self.ssd_hold
            self.pending_ssd = (effect['hold_earliest_ms'], payload, effect)
        else:
            self.dispatch_read(payload, effect)

    def dispatch_read(self, payload, effect):
        effect['native_read_call_ms'] = relative_ms(self.epoch)
        self.read_ticket, = self.wire.command(*payload)
        effect['fence'] = self.read_ticket

    def compute(self, action, now):
        self.wire.command('compute', action.request, action.boundary, int(action.release), int(action.shared_checkpoint))
        self.record('compute', now, action=asdict(action))
        return Launch(True)  # native ACK occurs only after suffix submission

    def poll(self):
        if self.pending_ssd and relative_ms(self.epoch) >= self.pending_ssd[0]:
            _, payload, effect = self.pending_ssd
            self.pending_ssd = None
            self.dispatch_read(payload, effect)
        ticket, done = self.wire.command('poll', tag='!poll')
        read = None
        if ticket:
            assert self.fifo.reading and ticket == self.read_ticket
            read = self.fifo.reading
            self.read_ticket = None
        return read, done if done >= 0 else None


def one_block(wire, args, rep, capacity):
    offsets = [int(i) for i in args.arrivals.split(',')]
    assert len(offsets) == 3 and offsets == sorted(offsets) and offsets[0] == 0
    exp = fixture([16777216, 52690944, 16777216, 52690944], capacity, offsets[-1], added_ssd_ms=args.ssd_hold_ms)
    # The catalogue uses these times for ordering only. The controller replaces
    # each value once that arrival is observed, before the chooser may see it.
    exp.arrivals = list(map(Fraction, offsets))
    start = perf_counter_ns()
    gpu = GPU(wire, start, args.ssd_hold_ms)
    submitted = SimpleQueue()
    stop = Event()

    def send():
        for i, t in enumerate(offsets):
            until = start + t * 1_000_000
            while not stop.is_set():
                delay = (until - perf_counter_ns()) / 1e9
                if delay <= 0 or stop.wait(delay):
                    break
            if stop.is_set():
                return
            submitted.put((i, relative_ms(start)))

    submitter = Thread(target=send, name='r10-open-loop', daemon=True)
    choices = []
    forces = [int(v) for v in args.force.split(',')] if args.force else None
    assert forces is None or len(forces) == 3

    def choose(exp, policy, snap):
        before = perf_counter_ns()
        result = choose_observed(exp, policy, snap) if forces is None else forces[snap.request]
        choices.append(dict(snapshot=asdict(snap), action=result, choose_ns=perf_counter_ns() - before))
        return result

    fifo = LiveFIFO(exp, args.policy, gpu, choose=choose)
    gpu.fifo = fifo
    arrived = []
    submitter.start()
    try:
        while not fifo.done:
            if relative_ms(start) > 20_000:
                raise RuntimeError('R10 short FIFO diagnostic exceeded the per-block deadline')
            landed, computed = gpu.poll() if fifo.active or fifo.reading else (None, None)
            actual = []
            while True:
                try:
                    actual.append(submitted.get_nowait())
                except Empty:
                    break
            if actual or landed or computed is not None:
                current = relative_ms(start)
                arrived.extend(actual)
                fifo.step(current, arrivals=[i for i, _ in actual], read_done=landed, compute_done=computed)
            else:
                sleep(.001)
    finally:
        stop.set()
        submitter.join(timeout=2)
    wire.command('finish', tag='!done')
    assert [i for i, _ in arrived] == list(range(3))
    return dict(rep=rep, policy=args.policy, force=forces, capacity_bytes=capacity, arrivals_planned_ms=offsets,
                controlled_ssd_hold_ms=args.ssd_hold_ms,
                arrivals_client_ms=[t for _, t in arrived], arrivals_observed_ms=fifo.exp.arrivals,
                completions_ms=fifo.completions, response_ms=[c - a for c, (_, a) in zip(fifo.completions, arrived, strict=True)],
                decisions=choices, peak_prefetch_bytes=fifo.peak_prefetch, effects=gpu.effects,
                transfers=[dict(read=asdict(r), start=s, end=e) for r, s, e in fifo.transfers])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for item in ('worker', 'runtime', 'model', 'scratch', 'output', 'observer'):
        parser.add_argument('--' + item, required=True, type=Path)
    parser.add_argument('--policy', choices=POLICIES, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--arrivals', default='0,35,70')
    parser.add_argument('--force', help='Control actions, e.g. 0,512,1024')
    parser.add_argument('--cold', action='store_true')
    parser.add_argument('--ssd-hold-ms', type=int, default=0, help='Controlled software hold before each real SSD read (0 is native)')
    args = parser.parse_args()
    assert args.repeats >= 1 and args.ssd_hold_ms >= 0 and not any(args.scratch.parent.glob(args.scratch.name + '.*'))
    env = os.environ | dict(PATH=str(args.runtime.resolve()) + os.pathsep + os.environ['PATH'],
                            GGML_VK_VISIBLE_DEVICES='1', GGML_VK_ASYNC_USE_TRANSFER_QUEUE='1',
                            GGML_VK_R10_CONCURRENT_BUFFERS='1', VK_INSTANCE_LAYERS='VK_LAYER_R10_timeline_observe',
                            VK_ADD_LAYER_PATH=str(args.observer.resolve().parent))
    process = [args.worker, args.model, args.runtime, 'Vulkan0', args.scratch, str(args.repeats),
               'fifo-link-cold' if args.cold else 'fifo-link', args.observer.resolve()]
    with (args.output.with_suffix('.csv').open('x', encoding='utf-8', newline='') as raw,
          args.output.with_suffix('.log').open('xb') as err,
          args.output.with_suffix('.jsonl').open('x', encoding='utf-8') as abslog,
          Popen(process, env=env, stdin=PIPE, stdout=PIPE, stderr=err, text=True, bufsize=1, encoding='ascii') as worker):
        wire = Wire(worker, raw)
        for i in range(-1, args.repeats):
            rep, capacity = wire.receive('!open')
            assert rep == i
            result = one_block(wire, args, rep, capacity)
            abslog.write(json.dumps(result, sort_keys=True, default=msjson) + '\n')
            abslog.flush()
        assert worker.wait(timeout=15) == 0 and not worker.stdout.read()


if __name__ == '__main__':
    main()
