"""Local HTTP/SSE transport for a known R10 replay block (no general inference API)."""

from codecs import getincrementaldecoder
from fractions import Fraction
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from queue import Empty, Queue
from secrets import token_hex
from threading import Lock, Thread
from time import perf_counter_ns


class LoopbackSSE:
    def __init__(self, n, submitted, timeout):
        self.n, self.submitted, self.timeout = n, submitted, timeout
        self.out = [Queue() for _ in range(n)]
        self.started, self.accepted, self.clients, self.error = {}, {}, {}, {}
        self.client_events, self.client_partial, self.errors_type = {}, {}, {}
        self.lock = Lock()
        self.next_to_publish = 0
        token = token_hex(16)
        self.prefix = f'/{token}/'
        active = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                tail = self.path.removeprefix(active.prefix) if self.path.startswith(active.prefix) else ''
                if not tail.isascii() or not tail.isdecimal() or len(tail) > 5 or int(tail) >= active.n:
                    self.send_error(404)
                    return
                i = int(tail)
                with active.lock:
                    valid = i in active.started and i not in active.accepted
                    if valid:
                        active.accepted[i] = perf_counter_ns()
                        while active.next_to_publish in active.accepted:
                            j = active.next_to_publish
                            active.submitted.put((j, active.ms(active.started[j])))
                            active.next_to_publish += 1
                if not valid:
                    self.send_error(409)
                    return
                try:
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    decode = getincrementaldecoder('utf8')(errors='replace')
                    while True:
                        item = active.out[i].get(timeout=active.timeout)
                        if item is None:
                            return
                        kind, data = item
                        if kind == 'token':
                            data = dict(data, text=decode.decode(bytes.fromhex(data['piece_hex'])))
                        else:
                            data = dict(data, text=decode.decode(b'', final=True))
                        encoded = json.dumps(data, ensure_ascii=True, separators=(',', ':'))
                        self.wfile.write(f'event: {kind}\ndata: {encoded}\n\n'.encode('ascii'))
                        self.wfile.flush()
                        if kind == 'end':
                            break
                except (Empty, OSError, ValueError) as error:
                    with active.lock:
                        active.error[i] = 'server: ' + str(error)
                        active.errors_type[i] = 'timeout' if isinstance(error, (Empty, TimeoutError)) else type(error).__name__

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = Thread(target=self.server.serve_forever, kwargs=dict(poll_interval=.05), daemon=True)
        self.thread.start()

    def ms(self, ns):
        return Fraction(ns - self.epoch, 1_000_000)

    def submit(self, i):
        connection = HTTPConnection('127.0.0.1', self.server.server_port, timeout=self.timeout)
        try:
            connection.connect()
            with self.lock:
                self.started[i] = perf_counter_ns()
            connection.request('POST', self.prefix + str(i), body=b'')
        except (OSError, ValueError) as error:
            connection.close()
            with self.lock:
                self.error[i] = 'submit: ' + str(error)
                self.errors_type[i] = 'timeout' if isinstance(error, TimeoutError) else type(error).__name__
            return False

        def read():
            records = []
            with self.lock:
                self.client_partial[i] = records
            try:
                with connection.getresponse() as response:
                    if response.status != 200 or not response.getheader('Content-Type', '').startswith('text/event-stream'):
                        raise ValueError(f'HTTP {response.status}; not an accepted SSE stream')
                    while True:
                        head = response.readline()
                        if not head.startswith(b'event: '):
                            raise ValueError('SSE stream ended without a terminal event')
                        event = head.decode('ascii').split()[1]
                        if event not in ('token', 'end'):
                            raise ValueError('unknown SSE event')
                        data = response.readline()
                        if not data.startswith(b'data: ') or response.readline() != b'\n':
                            raise ValueError('invalid single frame SSE response')
                        received = self.ms(perf_counter_ns())
                        row = dict(json.loads(data[6:]), event=event, client_received_ms=received)
                        with self.lock:
                            records.append(row)
                            if event == 'end':
                                self.client_events[i] = records
                        if event == 'end':
                            break
            except (OSError, ValueError) as error:
                with self.lock:
                    self.error[i] = 'client: ' + str(error)
                    self.errors_type[i] = 'timeout' if isinstance(error, TimeoutError) else type(error).__name__
            finally:
                connection.close()

        worker = Thread(target=read, daemon=True)
        self.clients[i] = worker
        worker.start()
        return True

    def send(self, i, event, data):
        self.out[i].put((event, data))

    def collect(self):
        maximum = perf_counter_ns() + 5_000_000_000
        for client in self.clients.values():
            client.join(max(0., (maximum - perf_counter_ns()) / 1e9))
        with self.lock:
            if self.error or len(self.client_events) != self.n:
                raise ValueError(f'loopback SSE incomplete, completed {len(self.client_events)} of {self.n}, errors {self.error}')
            return [dict(request=i, server_accept_ms=self.ms(self.accepted[i]), events=self.client_events[i]) for i in range(self.n)]

    def partial(self):
        with self.lock:
            epoch = getattr(self, 'epoch', None)
            snapshot = perf_counter_ns()
            rows = [dict(request=i, started_ms=self.ms(self.started[i]) if i in self.started else None,
                         accepted_ms=self.ms(self.accepted[i]) if i in self.accepted else None,
                         status=('terminal' if i in self.client_events else 'timed_out' if self.errors_type.get(i) == 'timeout'
                                 else 'client_error' if i in self.error else 'unfinished' if i in self.started else 'not_submitted'),
                         error_type=self.errors_type.get(i), error=self.error.get(i),
                         events=list(self.client_partial.get(i, ()))) for i in range(self.n)]
        return dict(requests=rows, observation_end_ms=self.ms(snapshot) if epoch is not None else None)

    def close(self):
        for stream in self.out:
            stream.put(None)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
