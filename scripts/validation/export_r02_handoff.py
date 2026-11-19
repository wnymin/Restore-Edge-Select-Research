"""Export the named R02 cross-review cases separately from their outcomes."""

import argparse
from fractions import Fraction
import json
from pathlib import Path

from scripts.gpu.experiments.study_r02_gpu_observation import calibration, predict


EVIDENCE = Path("docs/evidence")
GPU_RAW = EVIDENCE / "r02-gpu-observation-7941.json"
GPU_AUDIT = EVIDENCE / "r02-gpu-observation-7941-audit.json"
CPU_DEV = EVIDENCE / "r02-cpu-dev-release-sharing-v1.json"
GPU_IDS = tuple(f"R02-GPU-01/v1:{name}:13" for name in ("tight-short", "wide-short"))
CPU_IDS = tuple(f"R02-CPU-01/dev-release-sharing-v1:release-sharing-{name}-120"
                for name in ("distinct", "shared-kv"))


def read(path):
    return json.loads(path.read_bytes())


def cases():
    gpu = read(GPU_RAW)
    cfg = gpu["config"]
    fit = calibration(Path(cfg["calibration"]))
    audit = {r["case_id"]: r for r in read(GPU_AUDIT)["pairs"]}
    device = {(r["case_id"], r["arm"]): r["after"] for r in gpu["outcomes"]}
    capture = {(r["case_id"], r["arm"]): r for r in gpu["observations"]}
    cells = {r["id"]: r for r in cfg["cells"]}
    observations, outcomes, evaluations = [], [], []
    for cid in GPU_IDS:
        # Prediction uses the one common decision observation, not deep replay.
        source = capture[cid, "shallow"]
        cell = cells[source["cell"]]
        remaining, _ = predict(cell, source["before"], cfg, fit)
        for arm, size in (("shallow", 16), ("deep", 48)):
            clock = source["before"]
            key = dict(case_id=cid, arm=arm)
            observations.append(dict(**key, source=GPU_RAW.as_posix(), version=cfg["protocol"],
                split="gpu-holdout", seed=cfg["seed"], capacity_bytes=cell["quota_mib"] << 20,
                boundary_tokens=None, proxy_bytes=size << 20, shared_object_ids=[], queue_head="B",
                known_fifo=["B", "C", "D"], decision_tick=clock["query_bracket_ns"][1],
                timebase="host monotonic ns from origin-submit bracket start", observed=clock,
                remaining_ms=remaining, remaining_kind="window marker, not physical activity",
                missing={"boundary_tokens": "synthetic boundary proxy", "active_compute": "not observable",
                         "active_peer_transfer": "not observable"}))
            r = device[cid, arm]
            outcomes.append(dict(**key, timebase="CUDA elapsed ms from that arm's origin event",
                release_ms=r["release_ms"], peer_ready_ms={"D": r["peer_d_ready_ms"]},
                completion_ms=dict(B=r["target_done_ms"], C=r["peer_c_done_ms"], D=r["peer_d_done_ms"])))
        p = audit[cid]
        evaluations.append(dict(case_id=cid, truth=p["truth"], tie_ms=cfg["tie_ms"],
                                deep_minus_shallow_ms=p["observed_delta_ms"], picks=p["picks"]))

    cpu = read(CPU_DEV)
    lookup = {r["case_id"]: r for r in cpu["cases"]}
    for cid in CPU_IDS:
        row = lookup[cid]
        ids, state = row["request_ids"], row["initial_snapshot"]
        arms = sorted(row["arms"], key=int)
        # Extract identity from the fixed follower plans in the original input.
        successors = row["arms"][arms[0]]["plan"][2:]
        deps = [set(next(a["load_order"] for a in r["candidates"] if a["boundary_tokens"] == c))
                for r, c in zip(row["trace"]["requests"][2:], successors, strict=True)]
        for arm in arms:
            key = dict(case_id=cid, arm=arm)
            r = row["arms"][arm]
            active_end = state["compute_end_ms"]
            residual = str(Fraction(active_end) - Fraction(state["now_ms"])) if active_end is not None else "0"
            observations.append(dict(**key, source=CPU_DEV.as_posix(), version=cpu["protocol"],
                split="cpu-development-only", model=cpu["model_config"], capacity_bytes=row["capacity_bytes"],
                boundary_tokens=int(arm), proxy_bytes=None, shared_object_ids=sorted(set.intersection(*deps)),
                queue_head=ids[state["head_index"]], known_fifo=ids, fixed_plan=r["plan"],
                decision_tick=state["now_ms"], timebase="exact simulator ms", observed=state,
                remaining_ms=[residual, residual], remaining_kind="simulated active compute", missing={}))
            outcomes.append(dict(**key, timebase="exact simulator ms", release_ms=r["compute_start_ms"][1],
                prefetch_reserved_ms=dict(zip(ids, r["prefetch_reserved_ms"], strict=True)),
                completion_ms=dict(zip(ids, r["completion_ms"], strict=True)), transfers=r["events"]))
        evaluations.append(dict(case_id=cid, exact_minima=row["exact_minima"],
                                score_order_pick=row["score_order_pick"], original_fallback=row["original_fallback"]))
    return dict(interface="R02-case/v1", observations=observations, outcomes=outcomes, evaluations=evaluations)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", type=Path)
    action.add_argument("--check", type=Path)
    args = p.parse_args()
    export = cases()
    keys = [{(r["case_id"], r["arm"]) for r in export[k]} for k in ("observations", "outcomes")]
    if not len(keys[0]) == len(export["observations"]) == len(export["outcomes"]) or keys[0] != keys[1]:
        p.exit(1, "Invalid exchange keys\n")
    if args.write:
        with args.write.open("x", encoding="utf8") as f:
            f.write(json.dumps(export, indent=2) + "\n")
    elif export != read(args.check):
        p.exit(1, "R02 exchange evidence does not match source data\n")
    print(f"Checked {len(export['evaluations'])} R02 exchange cases, {len(keys[0])} paired arms")


if __name__ == "__main__":
    main()
