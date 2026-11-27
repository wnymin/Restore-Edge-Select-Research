"""Single-request suffix cost models; time is in milliseconds."""

from fractions import Fraction
from bisect import bisect_left
from statistics import median


def causal_pairs(cached, suffix):
    return cached * suffix + suffix * (suffix + 1) // 2


def sampled_cost(config):
    rows = config["samples"]
    xs = sorted({r["cached_tokens"] for r in rows})
    ys = sorted({0, *(r["suffix_tokens"] for r in rows)})
    table = {(r["cached_tokens"], r["suffix_tokens"]):
             median(Fraction(str(v)) for v in r["ms"]) for r in rows}
    if (not xs or xs[0] != 0 or len(ys) < 2 or ys[0] != 0
            or len(table) != len(rows) or len(table) != len(xs) * (len(ys) - 1)
            or any(s == 0 or v <= 0 for (_, s), v in table.items())):
        raise ValueError("Require a complete positive sample grid including cached_tokens=0")
    table.update({(x, 0): Fraction(0) for x in xs})

    def bracket(axis, value):
        i = bisect_left(axis, value)
        if i < len(axis) and axis[i] == value:
            return [(value, Fraction(1))]
        if i == 0 or i == len(axis):
            raise ValueError(f"Prefill shape outside measured grid: {value} not in {axis[0]}..{axis[-1]}")
        lo, hi = axis[i - 1:i + 1]
        weight = Fraction(value - lo, hi - lo)
        return [(lo, 1 - weight), (hi, weight)]

    def predict(cached, suffix):
        if not suffix:
            return Fraction(0)
        return sum(table[x, y] * a * b for x, a in bracket(xs, cached) for y, b in bracket(ys, suffix))

    return predict


def build_prefill_cost(config, spec):
    kind = config.get("kind", "polynomial")
    if kind == "sampled-prefill":
        return sampled_cost(config)
    if kind == "chunk-sampled-prefill":
        chunk = config["chunk_tokens"]
        if type(chunk) is not int or chunk <= 0:
            raise ValueError("chunk_tokens must be a positive integer")
        per_chunk = sampled_cost(config)

        def predict(cached, suffix):
            # Follow exactly the same physical splits as the FIFO worker.
            return sum((per_chunk(cached + at, min(chunk, suffix - at))
                        for at in range(0, suffix, chunk)), Fraction(0))

        return predict
    if kind == "polynomial":
        a = Fraction(*config["a_ms_per_token_rational"])
        b = Fraction(*config["b_ms_per_pair_rational"])
        if min(a, b) < 0:
            raise ValueError("Recompute coefficients must be nonnegative")
        return lambda cached, suffix: a * suffix + b * causal_pairs(cached, suffix)
    if kind != "hybrid-roofline":
        raise ValueError(f"Unknown recompute model: {kind}")

    dense_layers = len(spec["full_attention_layers_1based"])
    recurrent_layers = spec["num_layers"] - dense_layers
    heads = spec["full_attention"]
    pair_flops = 4 * heads["query_heads"] * heads["head_dim"]
    kv_bytes = spec["payload"]["attention_bytes_per_token"]
    # The initial model shares effective rates across components. Launch
    # and projection weight costs are per request.
    def param(key):
        value = Fraction(str(config[key]))
        if value < 0:
            raise ValueError(f"Negative recompute parameter: {key}")
        return value

    compute = param("effective_flops_per_ms")
    bandwidth = param("effective_bytes_per_ms")
    if not compute or not bandwidth:
        raise ValueError("Roofline rates must be positive")
    launch = param("launch_ms")
    weights = param("projection_weight_bytes")
    projection = param("projection_flops_per_token")
    recurrence = param("recurrent_flops_per_layer_token")
    recurrent_traffic = param("recurrent_bytes_per_layer_token")
    recurrent_fixed = param("recurrent_fixed_bytes_per_layer")

    def predict(cached, suffix):
        if not suffix:
            return Fraction(0)
        # Model one logical KV sweep, rather than charging repeated loads
        # per query. Calibrate effective rates against the actual kernel.
        attention_ms = max(dense_layers * pair_flops * causal_pairs(cached, suffix) / compute,
                           kv_bytes * (cached + suffix) / bandwidth)
        projection_ms = max(projection * suffix / compute, weights / bandwidth)
        recurrent_ms = recurrent_layers * max(recurrence * suffix / compute,
                                              (recurrent_fixed + recurrent_traffic * suffix) / bandwidth)
        return launch + projection_ms + attention_ms + recurrent_ms

    return predict
