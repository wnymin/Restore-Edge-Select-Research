"""Drive the variable native catalogue with the common FIFO controller."""

import argparse
from dataclasses import asdict
from contextlib import ExitStack
from datetime import datetime, timezone
import ctypes
from fractions import Fraction
import json
import os
from pathlib import Path
from queue import Empty, SimpleQueue
from subprocess import CREATE_NO_WINDOW, DEVNULL, PIPE, Popen, TimeoutExpired
from threading import Event, Thread
from time import perf_counter_ns, sleep

from restore_edge_select.live_fifo import Launch, LiveFIFO
from restore_edge_select.live_snapshot import choose_observed
from restore_edge_select.simulator import Experiment
from scripts.gpu.experiments import r10_holdout_lock
from scripts.gpu.experiments.run_r10_catalog_prepare import ERROR_MODE, VK_OPTIONS
from scripts.gpu.experiments.run_r10_fifo_link import relative_ms
from scripts.gpu.experiments.run_r10_online_diagnostic import MIB, msjson
from scripts.validation.check_r10_native_catalog import digest


class Pipe:
    def __init__(self, process, raw, timeout, service=None):
        self.process, self.raw, self.timeout = process, raw, timeout
        self.deadline_ns = None
        self.replies = SimpleQueue()
        def receive():
            try:
                for line in process.stdout:
                    if line.startswith('!'):
                        self.replies.put(line)
                    elif service is not None and line.startswith('~'):
                        service.receive(line, perf_counter_ns())
                    else:
                        raw.write(line)
            finally:
                self.replies.put(None)
        self.receiver = Thread(target=receive, name='r10-native-pipe', daemon=True)
        self.receiver.start()

    def response(self, tag):
        timeout = self.timeout
        if self.deadline_ns is not None:
            timeout = min(timeout, max(0, (self.deadline_ns - perf_counter_ns()) / 1e9))
        try:
            line = self.replies.get(timeout=timeout)
        except Empty:
            raise TimeoutError(f'Native worker did not respond to {tag}') from None
        if line is None:
            raise RuntimeError(f'Worker exited before {tag}; see native stderr')
        got, *fields = line.split()
        if got != tag:
            raise RuntimeError(f'Expected {tag}, got {got}')
        return tuple(map(int, fields))

    def command(self, *args, tag='!ok'):
        self.process.stdin.write(' '.join(map(str, args)) + '\n')
        self.process.stdin.flush()
        return self.response(tag)


class GPU:
    def __init__(self, pipe):
        self.pipe = pipe
        self.reading = None
        self.controller = None

    def reserve(self, request, deps, active_kv, now):
        self.pipe.command('reserve', request, self.controller.chosen[request], len(active_kv), *sorted(map(int, active_kv)))

    def read(self, action, now):
        if self.reading is not None:
            raise ValueError('native transport lane is occupied')
        self.pipe.command('read', action.request, self.controller.chosen[action.request], action.object_id, action.lane_use)
        self.reading = action

    def compute(self, action, now):
        done, = self.pipe.command('compute', action.request, action.boundary, int(action.release), int(action.shared_checkpoint))
        return Launch(True, bool(done))

    def poll(self):
        ticket, done = self.pipe.command('poll', tag='!poll')
        reading = None
        if ticket:
            if not self.reading or self.reading != self.controller.reading:
                raise ValueError('unexpected native transport completion')
            reading, self.reading = self.reading, None
        return reading, done if done >= 0 else None


def drive(pipe, experiment, policy, scale, timeout, publish_rest_on='open-loop', forced=None, service=None):
    gpu = GPU(pipe)
    decisions = []
    def choose(exp, p, snapshot):
        since = perf_counter_ns()
        if forced is not None:
            action = forced[snapshot.request]
        elif p == 'deepest':
            action = max(exp.eligible[snapshot.request])
        else:
            action = choose_observed(exp, p, snapshot)
        decisions.append(dict(snapshot=asdict(snapshot), action=action, choose_ns=perf_counter_ns() - since))
        return action
    fifo = LiveFIFO(experiment, policy, gpu, choose=choose)
    gpu.controller = fifo
    offsets = [(x - experiment.arrivals[0]) * scale for x in experiment.arrivals]
    submitted = SimpleQueue()
    stop = Event()
    submit_errors = []
    sse = None
    if service is not None:
        service.open(len(offsets), submitted, timeout)
        sse = service.sse
    epoch_utc = datetime.now(timezone.utc).isoformat()
    start = perf_counter_ns()
    if sse is not None:
        sse.epoch = start
    pipe.deadline_ns = start + timeout * 1_000_000_000
    actual = []

    def arrivals():
        for i, planned in enumerate(offsets):
            until = start + int(planned * 1_000_000)
            delay = max(0., (until - perf_counter_ns()) / 1e9)
            if stop.wait(delay):
                return
            if sse is None:
                submitted.put((i, relative_ms(start)))
            elif not sse.submit(i):
                submit_errors.append(i)
                return

    submitter = None
    if publish_rest_on == 'open-loop':
        submitter = Thread(target=arrivals, name='r10-open-loop', daemon=True)
        submitter.start()
    else:
        submitted.put((0, relative_ms(start)))
    triggered = False
    try:
        while not fifo.done:
            if submit_errors:
                raise ValueError(f'loopback client could not submit request {submit_errors[0]}')
            if relative_ms(start) > timeout * 1000:
                raise TimeoutError('native calibration FIFO exceeded its block deadline')
            anchor = (publish_rest_on == 'first-compute' and fifo.active and fifo.active.request == 0 or
                      publish_rest_on == 'first-ssd' and fifo.reading and fifo.reading.request == 0 and
                      experiment.objects[fifo.reading.object_id]['tier'] == 'SSD')
            if not triggered and anchor:
                # One controlled publication before polling the anchor. The real
                # completion still arrives through the device's normal channel.
                stamp = relative_ms(start)
                for i in range(1, len(offsets)):
                    submitted.put((i, stamp))
                triggered = True
                landed, completed = None, None
            else:
                landed, completed = gpu.poll() if fifo.active or fifo.reading else (None, None)
            current = []
            while True:
                try:
                    current.append(submitted.get_nowait())
                except Empty:
                    break
            if current or landed or completed is not None:
                actual.extend(current)
                fifo.step(relative_ms(start), arrivals=[i for i, _ in current], read_done=landed, compute_done=completed)
            else:
                sleep(.001)
    finally:
        drain_utc = datetime.now(timezone.utc).isoformat()
        stop.set()
        if submitter:
            submitter.join(timeout=2)
    pipe.command('finish', tag='!done')
    if [i for i, _ in actual] != list(range(len(offsets))) or gpu.reading is not None:
        raise ValueError('calibration run did not account for all requests and transfers')
    service_results = {} if service is None else dict(service=service.complete(start, len(offsets)))
    return dict(policy=policy, publish_rest_on=publish_rest_on,
                **service_results,
                arrival_epoch_utc=epoch_utc, drain_observed_utc=drain_utc,
                planned_arrival_offsets_ms=offsets if submitter else None, client_arrivals_ms=[x for _, x in actual],
                observed_arrivals_ms=fifo.exp.arrivals, completions_ms=fifo.completions, peak_prefetch_bytes=fifo.peak_prefetch,
                decisions=decisions, starts=[dict(compute=asdict(c), observed_ms=t) for c, t in fifo.starts],
                transfers=[dict(read=asdict(read), start=a, done=b) for read, a, b in fifo.transfers])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('worker', 'runtime', 'model', 'plan', 'schedule', 'catalogue', 'seal', 'observer', 'out'):
        ap.add_argument('--' + name, required=True, type=Path)
    ap.add_argument('--policy', choices=('local-cost', 'entry-replay', 'deepest', 'forced'), required=True)
    ap.add_argument('--forced-actions', type=Path, help='fixed boundary for every request in a synthetic mechanism block')
    ap.add_argument('--ssd-hold-ms', type=int, default=0, help='synthetic SSD handoff delay after the real read is ready')
    ap.add_argument('--service-tokens', type=int, help='opt-in greedy generation with a fixed per-request short output cap')
    ap.add_argument('--service-long-tokens', type=int, default=0, help='larger cap for request indices divisible by four')
    ap.add_argument('--service-sse', action='store_true', help='route local service arrivals and token text through HTTP/SSE over loopback')
    ap.add_argument('--device', default='Vulkan0')
    ap.add_argument('--device-index', type=int, default=1)
    ap.add_argument('--capacity-mib', type=Fraction, required=True)
    ap.add_argument('--arrival-scale', type=Fraction, required=True)
    ap.add_argument('--timeout-s', type=int, required=True)
    ap.add_argument('--verify', action='store_true', help='compare untimed fixture states byte for byte with independent recomputation')
    ap.add_argument('--publish-rest-on', choices=('open-loop', 'first-compute', 'first-ssd'), default='open-loop')
    ap.add_argument('--cost-profile', type=Path, help='provisional profile for calibration, frozen profile for primary holdout')
    ap.add_argument('--holdout-lock', type=Path, help='frozen primary execution lock; omitted for calibration')
    ap.add_argument('--round', type=int, help='registered primary pair round')
    ap.add_argument('--service-lock', type=Path, help='frozen and separate service holdout execution lock')
    ap.add_argument('--service-case', help='registered paired service case')
    ap.add_argument('--warmup', action='store_true', help='uniform untimed worker warmup')
    ap.add_argument('--warmup-grid', action='store_true', help='warm up a fixed set of synthetic prefill shapes in calibration only')
    ap.add_argument('--resources', action='store_true', help='sample native allocation stages and process memory')
    ap.add_argument('--quiet-events', action='store_true', help='disable native event rows but retain device completion signals')
    ap.add_argument('--chunk-events', action='store_true', help='diagnostic host intervals for each native prefill chunk')
    ap.add_argument('--pipeline-events', action='store_true', help='diagnostic host intervals for Vulkan pipeline creation calls')
    ap.add_argument('--monitor-wddm', action='store_true', help='read device memory counters in a separate process; resource diagnostics only')
    ap.add_argument('--allocate-only', action='store_true', help='initialize the real model and arena for a resource diagnostic only')
    a = ap.parse_args()
    stage = a.capacity_mib * MIB
    if stage <= 0 or stage % 65536 or a.arrival_scale <= 0 or a.timeout_s <= 0:
        ap.error('stage must be whole nonzero 64 KiB pages; scale and timeout must be positive')
    if a.verify and a.quiet_events:
        ap.error('state qualification cannot suppress evidence')
    if a.chunk_events and (a.quiet_events or a.allocate_only):
        ap.error('chunk diagnostics require an executed block with native event rows')
    if a.pipeline_events and not a.chunk_events:
        ap.error('pipeline diagnostics require native chunk event rows')
    if a.warmup_grid and not a.warmup:
        ap.error('warmup grid requires common warmup')
    if a.monitor_wddm and not a.resources:
        ap.error('monitor WDDM only alongside native resource snapshots')
    if a.allocate_only and not (a.resources and a.warmup and not a.verify):
        ap.error('allocation-only mode requires warmup and resources, without state verification')
    if a.service_tokens is not None:
        if not (1 <= a.service_tokens <= 256 and (a.service_long_tokens == 0 or a.service_tokens < a.service_long_tokens <= 256)):
            ap.error('service limits must be positive, at most 256, and short < long when both are supplied')
        if a.allocate_only or a.quiet_events or a.ssd_hold_ms or a.holdout_lock:
            ap.error('service qualification is separate from the primary prefill holdout and requires native events')
        if a.service_sse and a.publish_rest_on != 'open-loop':
            ap.error('SSE requires the open-loop client')
    elif a.service_long_tokens or a.service_sse:
        ap.error('service options require a short output cap')
    seal = json.loads(a.seal.read_bytes())
    if bool(a.forced_actions) != (a.policy == 'forced') or (a.forced_actions and (seal['family'] != 'synthetic' or a.allocate_only)):
        ap.error('forced actions require the forced policy and a synthetic executed block')
    if a.ssd_hold_ms and (not 1 <= a.ssd_hold_ms <= 1000 or not a.forced_actions or a.quiet_events):
        ap.error('SSD soft hold is available only with synthetic forced actions and event logging (1–1000 ms)')
    if a.publish_rest_on != 'open-loop' and not ((a.verify or a.forced_actions) and seal['family'] == 'synthetic'):
        ap.error('device-anchored arrivals are limited to synthetic qualification and forced mechanism controls')
    holdout, service_holdout = bool(a.holdout_lock), bool(a.service_lock)
    if holdout != (a.round is not None) or service_holdout != bool(a.service_case) or (holdout and service_holdout):
        raise ValueError('held-out timing requires exactly its lock and associated round or case')
    if (holdout and seal['split'] not in ('reserved', 'holdout') or
            service_holdout and seal['split'] not in ('reserved', 'service-holdout')):
        raise ValueError('block split does not match the timing phase')
    if not (holdout or service_holdout or seal['split'] in ('pilot', 'calibration')) or not seal.get('run_receipt_sha256') or \
            (digest(a.plan), digest(a.schedule)) != (seal['wire_sha256'], seal['schedule_sha256']):
        raise ValueError('requires a sealed catalogue with matching plan')
    if holdout:
        lock = r10_holdout_lock.load(a.holdout_lock)
        r10_holdout_lock.timing(lock, a.holdout_lock, seal, a)
    if service_holdout:
        from scripts.gpu.experiments import r10_service_lock

        lock = r10_service_lock.load(a.service_lock)
        r10_service_lock.timing(lock, a.service_lock, seal, a)
    if holdout or service_holdout:
        prep, key = a.catalogue.with_suffix('.run.json'), 'holdout_lock_sha256' if holdout else 'service_lock_sha256'
        if digest(prep) != seal['run_receipt_sha256'] or json.loads(prep.read_bytes()).get(key) != digest(a.holdout_lock or a.service_lock):
            raise ValueError('held-out preparation provenance does not match the frozen lock')
    for label, ext, key in [('objects', 'bin', 'sha256'), ('checkpoints', 'meta', 'metadata_sha256')]:
        for i, row in seal[label].items():
            if digest(a.catalogue / f'{i}.{ext}') != row[key]:
                raise ValueError(f'calibration catalogue artifact {i}.{ext} does not match its seal')
    sch = json.loads(a.schedule.read_bytes())
    if a.cost_profile:
        profile = json.loads(a.cost_profile.read_bytes())
        frozen = holdout or service_holdout
        expected = 'R10 frozen stationary costs for primary holdout' if frozen else 'R10 provisional isolated costs for calibration only'
        if profile['scope'] != expected or profile['model_sha256'] != (lock['environment']['model_sha256'] if frozen else digest(a.model)):
            raise ValueError('cost profile is not for this phase and model')
        sch['trace']['cost'] = profile['cost']
    experiment = Experiment(sch['trace'], sch['model'], a.capacity_mib)
    forced = None
    if a.forced_actions:
        control = json.loads(a.forced_actions.read_bytes())
        forced = control['boundaries']
        if control['schedule_sha256'] != seal['schedule_sha256'] or len(forced) != len(experiment.requests) or any(
                type(d) is not int or d not in experiment.eligible[i] for i, d in enumerate(forced)):
            raise ValueError('forced action control does not match the schedule or every legal boundary')
    env = os.environ.copy()
    env.update(VK_OPTIONS)
    env.pop('R10_VK_PROFILE_PIPELINES', None)
    if a.pipeline_events:
        env['R10_VK_PROFILE_PIPELINES'] = '1'
    runtime, observer = a.runtime.resolve(), a.observer.resolve()
    env.update(PATH=str(runtime) + os.pathsep + env.get('PATH', ''), GGML_VK_VISIBLE_DEVICES=str(a.device_index),
               VK_ADD_LAYER_PATH=str(observer.parent), VK_INSTANCE_LAYERS='VK_LAYER_R10_timeline_observe')
    argv = [str(a.worker.resolve()), str(a.model.resolve()), str(runtime), a.device, str(a.plan.resolve()),
            str(a.catalogue.resolve()), str(int(stage)), str(observer)]
    for value, flag in ((a.verify, '--verify'), (a.warmup, '--warmup'), (a.warmup_grid, '--warmup-grid'), (a.resources, '--resources'),
                        (a.quiet_events, '--quiet-events'), (a.allocate_only, '--allocate-only'), (a.chunk_events, '--chunk-events')):
        if value:
            argv.append(flag)
    if a.ssd_hold_ms:
        argv.append(f'--ssd-hold-ms={a.ssd_hold_ms}')
    service = None
    if a.service_tokens is not None:
        from scripts.gpu.experiments.r10_service_events import ServiceEvents

        service = ServiceEvents(a.service_tokens, a.service_long_tokens, a.service_sse)
        argv.append(f'--service={a.service_tokens}:{a.service_long_tokens}')
    paths = {k: a.out.with_suffix(ext) for k, ext in (('native', '.native.csv'), ('stderr', '.stderr'),
                                                     ('result', '.run.json'), ('failure', '.failure.json'))}
    if a.monitor_wddm:
        paths.update(wddm=a.out.with_suffix('.wddm.csv'), wddm_error=a.out.with_suffix('.wddm.stderr'))
    if any(p.exists() for p in paths.values()):
        raise FileExistsError('native diagnostic output already exists')
    start_utc = datetime.now(timezone.utc).isoformat()
    with ExitStack() as stack:
        log = stack.enter_context(paths['native'].open('x', encoding='ascii', newline='\n'))
        stderr = stack.enter_context(paths['stderr'].open('xb'))
        setmode = ctypes.windll.kernel32.SetErrorMode
        before = setmode(ERROR_MODE)
        try:
            child = stack.enter_context(Popen(argv, env=env, stdin=PIPE, stdout=PIPE, stderr=stderr, text=True,
                                             encoding='ascii', creationflags=CREATE_NO_WINDOW))
        finally:
            setmode(before)
        monitor, failure = None, None
        try:
            if a.monitor_wddm:
                source = Path(__file__).with_name('r10_follow_gpu_memory.ps1').resolve()
                out = stack.enter_context(paths['wddm'].open('xb'))
                err = stack.enter_context(paths['wddm_error'].open('xb'))
                monitor = stack.enter_context(Popen(['powershell.exe', '-NoProfile', '-NonInteractive', '-File', str(source),
                                                     '-WorkerId', str(child.pid)], stdin=DEVNULL, stdout=out, stderr=err,
                                                    creationflags=CREATE_NO_WINDOW))
            pipe = Pipe(child, log, a.timeout_s, service)
            try:
                n, cap, ram, pinned = pipe.response('!allocated' if a.allocate_only else '!open')
                if n != len(experiment.requests) or cap != stage:
                    raise ValueError('native configuration does not match the controller')
                result = {} if a.allocate_only else drive(pipe, experiment, a.policy, a.arrival_scale, a.timeout_s,
                                                           a.publish_rest_on, forced, service)
                if child.wait(timeout=10):
                    raise RuntimeError('native worker has nonzero exit status')
            finally:
                receiver = pipe.receiver
        except (OSError, ValueError, RuntimeError, TimeoutExpired) as error:
            failure = (type(error).__name__, str(error))
            raise
        finally:
            status = child.poll()
            if status is None:
                child.kill()
                child.wait()
            if monitor:
                try:
                    monitor.wait(8)
                except TimeoutExpired:
                    monitor.kill()
                    monitor.wait()
            if 'receiver' in locals():
                receiver.join(timeout=2)
            if service is not None:
                service.close()
            if failure:
                service_record = {} if service is None else dict(
                    expected_service_requests=len(experiment.requests),
                    partial_sse=service.sse.partial() if service.sse is not None else None)
                with paths['failure'].open('x', encoding='utf8', newline='\n') as out:
                    json.dump(dict(block=seal['block'], policy=a.policy, start_utc=start_utc,
                                   finish_utc=datetime.now(timezone.utc).isoformat(), worker_pid=child.pid,
                                   worker_return_code=child.returncode, worker_return_code_hex=f'0x{child.returncode & 0xffffffff:08x}',
                                   terminated_by_controller=status is None, error_type=failure[0], error=failure[1], **service_record),
                              out, sort_keys=True, default=msjson)
                    out.write('\n')
    scope = 'R10 primary local holdout arm' if holdout else 'calibration diagnostic with provisional cost/arrival configuration; no performance claim'
    if service is not None:
        scope = 'R10 single-slot generation; separate from primary prefill holdout'
    if service_holdout:
        scope = 'R10 local streaming service holdout arm'
    result.update(scope=scope, qualification=a.verify,
                  block=seal['block'], capacity_bytes=int(stage), pinned_bytes=pinned, resident_ram_bytes=ram,
                  warmup=a.warmup, warmup_grid=a.warmup_grid, resources=a.resources, native_events=not a.quiet_events, chunk_events=a.chunk_events,
                  pipeline_events=a.pipeline_events, observer_sha256=digest(a.observer),
                  allocation_only=a.allocate_only, worker_pid=child.pid, monitor_exit_code=monitor.returncode if monitor else None,
                  forced_actions_sha256=digest(a.forced_actions) if a.forced_actions else None,
                  ssd_hold_ms=a.ssd_hold_ms,
                  cost_profile_sha256=digest(a.cost_profile) if a.cost_profile else None,
                  seal_sha256=digest(a.seal), worker_sha256=digest(a.worker))
    if holdout:
        result.update(round=a.round, holdout_lock_sha256=digest(a.holdout_lock), schedule_sha256=digest(a.schedule))
    if service_holdout:
        result.update(service_case=a.service_case, service_lock_sha256=digest(a.service_lock), schedule_sha256=digest(a.schedule))
    with paths['result'].open('x', encoding='utf8', newline='\n') as out:
        json.dump(result, out, default=msjson, sort_keys=True)
        out.write('\n')
    if a.allocate_only:
        print(f'Allocated {cap} stage bytes; no timed requests.')
    else:
        print(f'Executed {n} FIFO requests; peak staged {result["peak_prefetch_bytes"]} of {cap} bytes.')


if __name__ == '__main__':
    main()
