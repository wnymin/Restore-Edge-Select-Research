"""Check the committed fixture and replay all hand-calculated cases."""

from copy import deepcopy
from fractions import Fraction
from itertools import product
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


sys.path.insert(0, str(ROOT / "src"))

from restore_edge_select.simulator import Experiment


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    fixture = json.loads((ROOT / "fixtures/phase1-hand-trace.json").read_text())
    spec = json.loads((ROOT / "configs/qwen3.5-4b.json").read_text())
    model = spec["models"][0]
    require(fixture["model"] == model["model_id"], "model mismatch")
    require(f"/{fixture['revision']}/" in model["config_url"], "revision mismatch")
    payload = model["payload"]
    objects = {obj["id"]: obj for obj in fixture["objects"]}
    require(len(objects) == len(fixture["objects"]), "duplicate object ID")
    block = fixture["memory"]["kv_block_tokens"]
    arrival = [req["arrival_ms"] for req in fixture["requests"]]
    choices = []
    for req in fixture["requests"]:
        choices.append(set())
        token_run, = req["input_token_runs"]
        for candidate in req["candidates"]:
            c = candidate["boundary_tokens"]
            choices[-1].add(c)
            deps = [objects[obj] for obj in candidate["load_order"]]
            if c == 0:
                require(not deps, f"{req['id']}: full recompute loads objects")
                continue
            require(c % block == 0 and c <= token_run["count"], "invalid boundary")
            require(len(deps) == c // block + 1, "wrong dependency count")
            for end, obj in enumerate(deps[:-1], 1):
                require(obj["kind"] == "attention_kv" and obj["tier"] == "RAM"
                        and obj["start_token_exclusive"] == (end - 1) * block
                        and obj["end_token_inclusive"] == end * block
                        and obj["bytes"] == block * payload["attention_bytes_per_token"],
                        f"invalid KV block: {obj['id']}")
            state = deps[-1]
            require(state["kind"] == "recurrent_and_conv_checkpoint"
                    and state["tier"] == "SSD" and state["boundary_tokens"] == c
                    and state["bytes"] == payload["checkpoint_bytes"], "invalid checkpoint")
            for obj in deps:
                expected = obj.get("end_token_inclusive", c)
                require(obj["prefix_token_runs"] == [
                    {"token_id": token_run["token_id"], "count": expected}
                ], f"prefix mismatch: {obj['id']}")
    combos = set(product(*choices))
    cases = fixture["expected_by_prefetch_mib"]
    require(set(map(int, cases)) == set(fixture["memory"]["prefetch_mib_cases"]),
            "budget cases do not match fixture settings")
    for budget, case in cases.items():
        exp = Experiment(fixture, model, int(budget))
        runs = case["runs"]
        require(len(runs) == len(combos)
                and {tuple(run["boundaries_tokens"]) for run in runs} == combos,
                f"invalid boundary combinations at {budget} MiB")
        for run in runs:
            actual = exp.run("fixed", boundaries=run["boundaries_tokens"])
            for key, expected in run.items():
                require(actual[key] == expected,
                        f"{budget} MiB, {run['boundaries_tokens']}, {key}: "
                        f"{actual[key]} != {expected}")
            finish = run["completion_ms"]
            require(len(finish) == len(arrival) and finish == sorted(finish)
                    and all(f >= a for f, a in zip(finish, arrival)), "invalid completion order")
            require(0 <= run["peak_prefetch_bytes"] <= int(budget) * fixture["units"]["bytes_per_mib"],
                    "invalid prefetch peak")
            require(run["response_sum_ms"] == sum(f - a for f, a in zip(finish, arrival)),
                    "response sum mismatch")
        require(exp.offline()["offline_optima"] == case["offline_optima"], "simulator optimum mismatch")
        best = min(run["response_sum_ms"] for run in runs)
        require({tuple(run["boundaries_tokens"]) for run in runs if run["response_sum_ms"] == best}
                == set(map(tuple, case["offline_optima"])), "offline optimum mismatch")
    exp = Experiment(fixture, model, 160)
    require(exp.run("local-cost")["response_mean_ms"] == 31, "local-cost baseline mismatch")
    require(exp.run("capacity")["response_mean_ms"] == Fraction(123, 4), "capacity baseline mismatch")
    shared = deepcopy(fixture)
    b, c = shared["requests"][1:]
    c["input_token_runs"] = b["input_token_runs"]
    c["candidates"] = b["candidates"]
    shared["cost"]["device_copy"] = {
        "startup_ms": 0, "bytes_per_ms": 2 * payload["checkpoint_bytes"],
    }
    actual = Experiment(shared, model, 128).run("fixed", boundaries=[0, 576, 576])
    require(actual["completion_ms"] == [30, Fraction(139, 4), 39], "shared handoff timing")
    require(actual["peak_prefetch_bytes"] == 70385664, "shared reservation counted twice")
    require(actual["transferred_bytes_by_path"] == {
        "ram_to_device": 18874368, "ssd_to_device": 51511296, "device_copy": 51511296,
    }, "shared dependencies reloaded or checkpoint copy missing")
    print(f"Checked {len(objects)} objects and {len(cases) * len(combos)} acceptance cases.")


if __name__ == "__main__":
    main()
