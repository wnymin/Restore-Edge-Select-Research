"""Summarize observable prefix reuse and arrival gaps in a Bailian JSONL window."""

import argparse
from decimal import Decimal
from itertools import islice
import json
from pathlib import Path


def percentile(values, percent):
    return sorted(values)[(len(values) * percent + 99) // 100 - 1] if values else None


def inspect(rows, checkpoint_tokens):
    nodes = {}
    lengths, reused, gaps = [], [], []
    previous = None
    for row in rows:
        timestamp = Decimal(str(row["timestamp"]))
        if previous is not None:
            if timestamp < previous:
                raise ValueError("Arrival timestamps must be nondecreasing")
            gaps.append(float((timestamp - previous) * 1000))
        previous = timestamp
        length = row["input_length"]
        # Only complete blocks are observable. Parent-aware keys are safe
        # whether the upstream IDs are per-block or already cumulative.
        blocks = row["hash_ids"][:length // 16]
        if len(blocks) != length // 16:
            raise ValueError("Missing hashes for complete input blocks")
        parent = shared = 0
        matched = True
        for index, block in enumerate(blocks, 1):
            key = (parent, block)
            if key not in nodes:
                matched = False
                nodes[key] = len(nodes) + 1
            parent = nodes[key]
            if matched:
                shared = index * 16
        lengths.append(length)
        reused.append(shared)
    if not lengths:
        raise ValueError("No requests in selected window")
    return {
        "requests": len(lengths),
        "input_tokens_p95": percentile(lengths, 95),
        "arrival_gap_ms_p50": percentile(gaps, 50),
        "arrival_gap_ms_p95": percentile(gaps, 95),
        "shared_prefix_tokens_p95": percentile(reused, 95),
        "requests_with_shared_blocks": sum(x > 0 for x in reused),
        "checkpoint_stride_tokens": checkpoint_tokens,
        "requests_with_two_nonzero_candidate_positions": sum(x >= 2 * checkpoint_tokens for x in reused),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--checkpoint-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.start < 0 or args.limit <= 0 or args.checkpoint_tokens <= 0 or args.checkpoint_tokens % 16:
        parser.error("Require a nonnegative start, positive limit and positive 16-token checkpoint multiple")
    with args.trace.open(encoding="utf-8") as source:
        rows = (json.loads(line, parse_float=Decimal)
                for line in islice(source, args.start, args.start + args.limit))
        print(json.dumps(inspect(rows, args.checkpoint_tokens), indent=2))


if __name__ == "__main__":
    main()
