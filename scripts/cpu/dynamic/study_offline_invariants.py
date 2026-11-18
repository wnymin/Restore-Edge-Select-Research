"""Stress certified offline bounds against exhaustive small exact replays."""

import argparse
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
from random import Random

from scripts.cpu.dynamic.dynamic_cpu_inputs import FIXTURE, MODEL, ROOT, read, sha256
from scripts.cpu.replay.study_bailian import Experiment
from scripts.cpu.dynamic.study_dynamic_cpu_holdout import write_csv
from restore_edge_select.offline_bounds import search


def cases(seed, count):
    rng = Random(seed)
    fixture = read(FIXTURE)
    for number in range(count):
        data = deepcopy(fixture)
        shared = rng.choice((False, True))
        pool = data["requests"][1:]
        requests = [data["requests"][0]]
        now = F(0)
        for i in range(rng.choice((2, 3, 4))):
            request = deepcopy(pool[0 if shared else i % 2])
            now += rng.choice((F(0), F(1, 3), F(1), F(12), F(50)))
            requests.append(dict(request, id=f"R{i}", arrival_ms=str(now)))
        data["requests"] = requests
        if rng.choice((False, True)):
            rng.shuffle(data["requests"])
        release = rng.choice(("start", "finish", "timer"))
        data["scheduling"].update(
            future_window_requests=rng.choice((0, 1, 2, 4)), prefetch_release=release,
            prefetch_hold_ms=str(rng.choice((F(1, 3), F(1), F(20)))) if release == "timer" else 0,
            initial_device_object_ids=[obj["id"] for obj in data["objects"] if rng.randrange(8) == 0],
        )
        data["cost"]["device_copy"] = dict(startup_ms="1/10", bytes_per_ms=rng.choice((10000000, 100000000)))
        for path in ("ram_to_device", "ssd_to_device"):
            cfg = data["cost"][path]
            rate = rng.choice((F(1, 2), F(1), F(4), F(16)))
            cfg["startup_ms"] = str(rng.choice((F(0), F(1, 7), F(2))))
            cfg["bytes_per_ms"] = str(F(cfg["bytes_per_ms"]) / rate)
        overlap = rng.choice((False, True))
        if overlap:
            data["overlap_slowdown"] = {
                path: dict(compute=str(rng.choice((F(1), F(5, 4), F(2)))),
                           transfer=str(rng.choice((F(1), F(3, 2), F(2)))))
                for path in ("ram_to_device", "ssd_to_device")}
        budget = rng.choice((0, 70, 128, 160, 192))
        yield number, shared, overlap, budget, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--cases", type=int, default=240)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("Case count must be positive")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model = read(MODEL)["models"][0]
    records = []
    for number, shared, overlap, budget, data in cases(args.seed, args.cases):
        try:
            exp = Experiment(data, model, budget)
            assert exp.arrivals == sorted(exp.arrivals)
            optimum = exp.offline()["response_sum_ms"]
            feasible = [exp.run("local-cost")]
            for mode, options in (("plain", {}), ("resource", dict(resource_floor=True)),
                                  ("dominance", dict(dominance=True)),
                                  ("combined", dict(resource_floor=True, dominance=True))):
                previous_lo, previous_hi = F(0), feasible[0]["response_sum_ms"]
                for limit in (1, 8, 1000):
                    diagnostics = {}
                    lo, hi, plan, nodes = search(exp, feasible, limit, diagnostics=diagnostics, **options)
                    assert previous_lo <= lo <= optimum <= hi <= previous_hi
                    assert exp.run("fixed", boundaries=plan)["response_sum_ms"] == hi
                    if limit == 1000:
                        assert lo == hi == optimum
                    records.append(dict(
                        case=number, mode=mode, node_budget=limit, nodes=nodes,
                        requests=len(exp.requests), prefix_layout="single" if shared else "alternating",
                        overlap=int(overlap), budget=budget,
                        window=exp.window, release=exp.release_at, initial_objects=len(exp.initial),
                        lower_sum_ms=str(lo), optimum_sum_ms=str(optimum), upper_sum_ms=str(hi),
                        dominated=diagnostics["dominated"], exact=int(lo == hi),
                    ))
                    previous_lo, previous_hi = lo, hi
        except Exception:
            args.out.with_suffix(".failure.json").write_text(
                json.dumps(dict(seed=args.seed, case=number, budget=budget, trace=data),
                           indent=2, default=str) + "\n", encoding="utf-8")
            raise
        if (number + 1) % 40 == 0:
            print(f"Checked {number + 1} exact-clock configurations", flush=True)
    write_csv(args.out, records)
    args.out.with_suffix(".json").write_text(json.dumps(dict(
        seed=args.seed, cases=args.cases, modes=4, node_budgets=[1, 8, 1000],
        source_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in (
            Path(__file__).resolve(), FIXTURE, MODEL, ROOT / "src/restore_edge_select/simulator.py",
            ROOT / "src/restore_edge_select/offline_bounds.py")},
    ), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
