"""Check that the R10 reconstructed manifests preserve observable source structure."""

import argparse
from collections import Counter
from decimal import Decimal
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path

from scripts.data.build_r10_bailian import BLOCK, DEPTHS, HISTORY, PROMPT_MAX, ROOT, WINDOW, past_windows, sha


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def decode_lines(path):
    return [json.loads(line, parse_float=Decimal) for line in path.read_bytes().splitlines()]


def audit(manifest_path, reconstructed_dir):
    manifest = json.loads(manifest_path.read_bytes())
    seen_sources, equivalence_ids = {}, {}
    prefix_nodes = 0
    old_windows = past_windows()
    outputs = []
    for case in manifest["cases"]:
        source = case["source"]
        first, last = case["raw_rows"]
        require(first == WINDOW * case["window"] and last - first == WINDOW, case)
        require(all(abs(case["window"] - old) >= 4 for trace, old in old_windows if trace == source), case)
        used = seen_sources.setdefault(source, {})
        require(all(abs(case["window"] - prev) >= 4 for prev in used), case)
        used[case["window"]] = case["split"]
        source_path = ROOT / "data/raw/bailian" / source
        if len(used) == 1:
            require(sha(source_path) == case["source_sha256"], source)
        target = reconstructed_dir / case["generated_filename"]
        require(sha(target) == case["generated_sha256"], target)
        with source_path.open("rb") as f:
            section = list(islice(f, first, last))
        require(len(section) == WINDOW and sha256(b"".join(section)).hexdigest() == case["raw_window_sha256"], case)
        rows = [json.loads(line, parse_float=Decimal) for line in section]
        emitted = decode_lines(target)
        require(len(emitted) == case["requests"] == WINDOW - HISTORY, target)
        start = Decimal(str(rows[HISTORY]["timestamp"]))
        history = {tuple(hist["hash_ids"][:d // BLOCK]) for hist in rows[:HISTORY] for d in DEPTHS
                   if hist["input_length"] >= d and len(hist["hash_ids"]) >= d // BLOCK}
        # Walk in source and token tries together. Nodes maintain a per-parent bijection:
        # all pairs of prompts have identical complete-block longest common prefixes.
        trie = {0: ({}, {})}
        checks, truncations, internal_partial_tail = 0, 0, 0
        previous = Decimal(0)
        types = Counter()
        for rid, (row, request) in enumerate(zip(rows[HISTORY:], emitted, strict=True)):
            where = (source, case["window"], rid)
            arrival = Decimal(request["arrival_ms"])
            require(rid == request["request"] and arrival == (Decimal(str(row["timestamp"])) - start) * 1000
                    and previous <= arrival, where)
            previous = arrival
            require(row["input_length"] == request["source_input_length"] and
                    row["output_length"] == request["source_output_length"], where)
            tokens = request["tokens"]
            length = min(PROMPT_MAX, row["input_length"])
            require(len(tokens) == length and all(type(t) is int and 1000 <= t < 15000 for t in tokens), where)
            truncations += length < row["input_length"]
            types[str(row.get("type", "unlabeled"))] += 1
            internal_partial_tail += bool(length % BLOCK)
            hashes = tuple(row["hash_ids"][:length // BLOCK])
            require(len(hashes) == length // BLOCK, where)
            node = 0
            identities = {}
            for j, block_hash in enumerate(hashes, 1):
                chunk = tuple(tokens[(j - 1) * BLOCK:j * BLOCK])
                out, reverse = trie[node]
                if block_hash not in out:
                    out[block_hash] = (chunk, len(trie))
                    trie[len(trie)] = ({}, {})
                actual_chunk, node = out[block_hash]
                require(actual_chunk == chunk and reverse.setdefault(chunk, block_hash) == block_hash, where)
                if j * BLOCK in DEPTHS and hashes[:j] in history:
                    identities[j * BLOCK] = hashes[:j]
            candidates = request["checkpoints"]
            require([c["depth"] for c in candidates] == list(identities), where)
            for c in candidates:
                key = (source, c["depth"], identities[c["depth"]])
                actual = (c["prefix_id"], tuple(tokens[:c["depth"]]))
                require(len(c["prefix_id"]) == 64 and all(x in "0123456789abcdef" for x in c["prefix_id"])
                        and equivalence_ids.setdefault(key, actual) == actual, where)
                checks += 1
        prefix_nodes += len(trie) - 1
        require(truncations == case["truncated"], case)
        outputs.append(dict(source=source, window=case["window"], split=case["split"], requests=len(emitted),
                            checkpoint_references=checks, source_types=dict(types), truncated=truncations,
                            reconstructed_partial_tails=internal_partial_tail, complete_block_trie_nodes=len(trie) - 1))
    require(len(seen_sources) == 4, "four sources required")
    for windows in seen_sources.values():
        require(Counter(windows.values()) == {"pilot": 1, "reserved": 2}, "invalid split counts")
    return dict(input_manifest_sha256=sha(manifest_path), verified_cases=outputs,
                verified_requests=sum(o["requests"] for o in outputs), complete_block_trie_nodes=prefix_nodes,
                independent_text_workload_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("reconstructed", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.manifest, args.reconstructed)
    with args.out.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
