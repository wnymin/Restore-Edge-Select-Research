"""Build history-only cache snapshots for replaying public Bailian arrival windows."""

from fractions import Fraction as F
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from restore_edge_select.simulator import Experiment

BLOCK = 512


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build(rows, history, fixture, model, spacing, placement="split", max_prompt=None):
    nodes, cached, objects, requests = {}, set(), {}, []
    start = F(rows[history]["timestamp"])
    kv = model["payload"]["attention_bytes_per_token"]
    checkpoint = model["payload"]["checkpoint_bytes"]
    for i, row in enumerate(rows):
        length = row["input_length"]
        if i >= history and max_prompt is not None and length > max_prompt:
            continue
        if not 0 < length <= model["context_tokens"]:
            raise ValueError(f"Prompt outside modeled context: {length}")
        hashes = row["hash_ids"]
        if len(hashes) < length // 16:
            raise ValueError("Missing hashes for complete input blocks")
        runs, path = [], []
        parent = 0
        for j in range(length // BLOCK):
            key = (parent, tuple(hashes[j * 32:(j + 1) * 32]))
            parent = nodes.setdefault(key, len(nodes) + 1)
            path.append(parent)
            runs.append({"token_id": parent, "count": BLOCK})
        if i < history:
            cached.update(path)
            continue
        if length % BLOCK:
            runs.append({"token_id": -i - 1, "count": length % BLOCK})
        count = sum(node in cached for node in path)
        assert all(node in cached for node in path[:count])
        choices = [{"boundary_tokens": 0, "load_order": []}]
        deps = []
        for j, node in enumerate(path[:count], 1):
            obj = f"kv-{node}"
            deps.append(obj)
            objects.setdefault(obj, dict(id=obj, kind="attention_kv", tier="RAM",
                                         start_token_exclusive=(j - 1) * BLOCK,
                                         end_token_inclusive=j * BLOCK,
                                         prefix_token_runs=runs[:j], bytes=kv * BLOCK))
            # Retain at most two nonzero choices to isolate boundary selection.
            if j not in {1, count}:
                continue
            state = f"state-{node}"
            objects.setdefault(state, dict(id=state, kind="recurrent_and_conv_checkpoint",
                                           tier={"split": "RAM" if j == 1 else "SSD",
                                                 "ram": "RAM", "ssd": "SSD"}[placement],
                                           boundary_tokens=j * BLOCK,
                                           prefix_token_runs=runs[:j], bytes=checkpoint))
            choices.append(dict(boundary_tokens=j * BLOCK, load_order=[*deps, state]))
        requests.append(dict(id=str(i), arrival_ms=(F(row["timestamp"]) - start) * 1000 * spacing,
                             input_token_runs=runs, candidates=choices))
    if not requests:
        raise ValueError("Window has no evaluated requests within the prompt limit")
    longest = max(sum(run["count"] for run in r["input_token_runs"]) for r in requests)
    execution = (longest * kv + checkpoint + 2**20 - 1) // 2**20
    memory = dict(fixture["memory"], kv_block_tokens=BLOCK, execution_mib=execution + 32)
    memory["device_mib"] = memory["fixed_mib"] + memory["execution_mib"] + 384
    return dict(fixture, memory=memory, objects=list(objects.values()), requests=requests)


def p95(values):
    return sorted(values)[(95 * len(values) + 99) // 100 - 1]
