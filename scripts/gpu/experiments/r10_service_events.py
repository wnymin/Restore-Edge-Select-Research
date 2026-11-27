"""Validate opt-in native streaming frames received by the parent pipe client."""

from fractions import Fraction


class ServiceEvents:
    def __init__(self, short_limit, long_limit, serve_sse=False):
        self.short = short_limit
        self.long = long_limit
        self.frames = []
        self.wants_sse = serve_sse
        self.sse = None
        self.bad_frame = None

    def open(self, n, submitted, timeout):
        if self.wants_sse:
            from scripts.gpu.experiments.r10_service_sse import LoopbackSSE

            self.sse = LoopbackSSE(n, submitted, timeout)

    def close(self):
        if self.sse is not None:
            self.sse.close()

    @staticmethod
    def parse(line):
        event, req, *payload, native = line.split()
        piece = None
        if event == '~token' and len(payload) == 3:
            piece = payload.pop()
            piece = '' if piece == '-' else piece
            bytes.fromhex(piece)
        return event, int(req), [int(p) for p in payload], int(native), piece

    def receive(self, line, stamp):
        # This runs on the sole pipe reader, before the next command reply.
        self.frames.append((line, stamp))
        if self.sse is not None:
            try:
                event, req, payload, native, piece = self.parse(line)
                if event == '~token' and piece is not None:
                    index, token = payload
                    self.sse.send(req, 'token', dict(index=index, id=token, piece_hex=piece, native_us=native))
                elif event == '~end' and payload in ([0], [1]):
                    self.sse.send(req, 'end', dict(reason='eog' if payload[0] else 'length', native_us=native))
                elif event != '~prefill':
                    raise ValueError('SSE requires token pieces and a terminal frame')
            except (IndexError, ValueError) as error:
                self.bad_frame = str(error)

    def complete(self, epoch, requests):
        result = []
        current = None
        last_native = last_stamp = 0
        for line, stamp in self.frames:
            event, req, payload, native, piece = self.parse(line)
            if req != len(result) or native < last_native or stamp < last_stamp:
                raise ValueError('out-of-order service frame')
            last_native, last_stamp = native, stamp
            data = dict(native_us=native, parent_observed_ms=Fraction(stamp - epoch, 1_000_000))
            if event == '~prefill' and current is None and not payload:
                cap = self.long if self.long and req % 4 == 0 else self.short
                current = dict(request=req, limit=cap, prefill_done=data, tokens=[])
            elif event == '~token' and current is not None and len(payload) == 2:
                index, token = payload
                if index != len(current['tokens']) + 1 or index > current['limit'] or token < 0:
                    raise ValueError('invalid token sequence or output cap')
                current['tokens'].append(dict(data, index=index, id=token, piece_hex=piece))
            elif event == '~end' and current is not None and payload in ([0], [1]):
                n = len(current['tokens'])
                if (payload[0] == 0 and n != current['limit']) or (payload[0] == 1 and n >= current['limit']):
                    raise ValueError('service ended inconsistent with its output cap')
                current.update(finish=dict(data, reason='eog' if payload[0] else 'length'))
                result.append(current)
                current = None
            else:
                raise ValueError(f'unexpected native service frame: {event}')
        if current is not None or len(result) != requests:
            raise ValueError('native output did not account for every client request')
        output = dict(scope='one compute slot, real greedy token IDs through a local process pipe',
                      short_limit=self.short, long_limit=self.long, requests=result)
        if self.sse is not None:
            if self.bad_frame:
                raise ValueError('malformed streaming frame: ' + self.bad_frame)
            output.update(scope='one compute slot, real greedy output delivered as SSE to a loopback HTTP client',
                          http_sse_requests=self.sse.collect())
        return output
