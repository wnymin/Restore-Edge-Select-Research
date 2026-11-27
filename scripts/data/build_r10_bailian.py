"""Prepare R10 Bailian token inputs without using policy results to pick windows."""

import argparse
from decimal import Decimal
from hashlib import file_digest, sha256
from itertools import islice
import json
from pathlib import Path
from struct import unpack

from scripts.validation.run_r08_sealed_holdout import excluded

ROOT = Path(__file__).resolve().parents[2]
WINDOW = 352
HISTORY = 256
BLOCK = 16
PROMPT_MAX = 2560
DEPTHS = (512, 1024, 1536)
SALT = "r10-reconstructed-load-v1"


def sha(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def windows(path):
    with path.open("rb") as f:
        while lines := list(islice(f, WINDOW)):
            if len(lines) < WINDOW:
                return
            yield lines


def row_blocks(row):
    length = min(row["input_length"], PROMPT_MAX)
    hashes = row["hash_ids"][:length // BLOCK]
    if length < 1 or len(hashes) != length // BLOCK:
        raise ValueError("Invalid length or missing complete prompt hash blocks")
    return length, tuple(hashes)


def token_block(prefix):
    # Token-ID input uses IDs from the range already exercised by the 4B probe.
    return [1000 + n % 14000 for n in unpack("<16H", prefix)]


def rebuild(lines):
    rows = [json.loads(line, parse_float=Decimal) for line in lines]
    request_rows = rows[HISTORY:]
    first = Decimal(str(request_rows[0]["timestamp"]))
    if any(b["timestamp"] < a["timestamp"] for a, b in zip(rows, rows[1:])):
        raise ValueError("Source arrival timestamps not monotonic")

    prefixes = set()
    for row in rows[:HISTORY]:
        _, hashes = row_blocks(row)
        prefixes.update(hashes[:depth // BLOCK] for depth in DEPTHS if len(hashes) >= depth // BLOCK)

    requests, checkpoint_ids, generated = [], set(), {}
    for i, row in enumerate(request_rows):
        length, hashes = row_blocks(row)
        prefix = b""
        tokens, checkpoints = [], []
        for j, block_hash in enumerate(hashes, 1):
            prefix = sha256(prefix + json.dumps(block_hash, separators=(",", ":")).encode()).digest()
            block = token_block(prefix)
            identity = tuple(block)
            if generated.setdefault(identity, prefix) != prefix:
                raise ValueError("Unexpected token collision between distinct source prefix blocks")
            tokens.extend(block)
            depth = j * BLOCK
            if depth in DEPTHS and hashes[:j] in prefixes:
                checkpoints.append(dict(depth=depth, prefix_id=prefix.hex()))
                checkpoint_ids.add((depth, prefix.hex()))
        if len(tokens) < length:
            tail = sha256(f"{SALT}/tail/{i}".encode()).digest()
            tokens.extend(token_block(tail)[:length - len(tokens)])
        requests.append(dict(request=i, arrival_ms=str((Decimal(str(row["timestamp"])) - first) * 1000),
                             source_input_length=row["input_length"], source_output_length=row["output_length"],
                             tokens=tokens, checkpoints=checkpoints))
    return requests, dict(requests=len(requests), unique_checkpoints=len(checkpoint_ids),
                          truncated=sum(len(r["tokens"]) < r["source_input_length"] for r in requests),
                          with_checkpoints=sum(bool(r["checkpoints"]) for r in requests),
                          with_multiple_checkpoints=sum(len(r["checkpoints"]) > 1 for r in requests))


def past_windows():
    used = excluded()
    for filename, sections in (("r08-takeover-seal.json", ("cases",)),
                               ("r09-inputs.json", ("calibration", "holdout"))):
        data = json.loads((ROOT / "docs/evidence" / filename).read_bytes())
        for section in sections:
            for r in data[section]:
                if "trace" in r and "window" in r:
                    used.add((r["trace"], r["window"]))
    return used


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    args = p.parse_args()
    used = past_windows()
    manifest = json.loads((ROOT / "data/bailian-manifest.json").read_bytes())
    config = dict(window_rows=WINDOW, history_rows=HISTORY, prompt_max=PROMPT_MAX,
                  candidate_depths=DEPTHS, selection_salt=SALT, previous_ids_checked=len(used),
                  workload="Bailian arrivals and prefix hashes reconstructed as token IDs",
                  selection="SHA-256 ordering of unused windows separated by >=4 window indices", cases=[])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for source in manifest["files"]:
        name = source["name"]
        path = ROOT / "data/raw/bailian" / name
        if path.stat().st_size != source["bytes"] or sha(path) != source["sha256"]:
            raise ValueError(f"Pinned source differs: {path}")
        prior = {i for t, i in used if t == name}
        candidates = [i for i, _ in enumerate(windows(path)) if all(abs(i - j) >= 4 for j in prior)]
        rank = sorted(candidates, key=lambda i: sha256(f"{SALT}:{name}:{i}".encode()).digest())
        chosen = []
        for index in rank:
            if all(abs(index - j) >= 4 for j in chosen):
                chosen.append(index)
            if len(chosen) == 3:
                break
        if len(chosen) != 3:
            raise ValueError(f"Insufficient source windows for {name}")
        for i, lines in enumerate(windows(path)):
            if i not in chosen:
                continue
            requests, statistics = rebuild(lines)
            label = "pilot" if i == chosen[0] else "reserved"
            target = args.out_dir / f"{path.stem}-{i}-{label}.jsonl"
            with target.open("x", encoding="utf-8", newline="\n") as out:
                for request in requests:
                    out.write(json.dumps(request, separators=(",", ":")) + "\n")
            config["cases"].append(dict(source=name, source_sha256=source["sha256"], window=i, split=label,
                                        raw_rows=[i * WINDOW, (i + 1) * WINDOW],
                                        raw_window_sha256=sha256(b"".join(lines)).hexdigest(),
                                        generated_filename=target.name, generated_sha256=sha(target), **statistics))
    with args.summary.open("x", encoding="utf-8") as out:
        json.dump(config, out, indent=2)
        out.write("\n")


if __name__ == "__main__":
    main()
