"""Check the committed acceptance fixture without running the simulator."""

from itertools import product
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
        runs = case["runs"]
        require(len(runs) == len(combos)
                and {tuple(run["boundaries_tokens"]) for run in runs} == combos,
                f"invalid boundary combinations at {budget} MiB")
        for run in runs:
            finish = run["completion_ms"]
            require(len(finish) == len(arrival) and finish == sorted(finish)
                    and all(f >= a for f, a in zip(finish, arrival)), "invalid completion order")
            require(0 <= run["peak_prefetch_bytes"] <= int(budget) * fixture["units"]["bytes_per_mib"],
                    "invalid prefetch peak")
            require(run["response_sum_ms"] == sum(f - a for f, a in zip(finish, arrival)),
                    "response sum mismatch")
        best = min(run["response_sum_ms"] for run in runs)
        require({tuple(run["boundaries_tokens"]) for run in runs if run["response_sum_ms"] == best}
                == set(map(tuple, case["offline_optima"])), "offline optimum mismatch")
    print(f"Checked {len(objects)} objects and {len(cases) * len(combos)} acceptance cases.")


if __name__ == "__main__":
    main()
