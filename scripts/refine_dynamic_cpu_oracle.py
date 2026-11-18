"""Refine the widest archived offline intervals without changing online rules."""

import argparse
import csv
from fractions import Fraction as F
import json
from pathlib import Path
from time import perf_counter

from dynamic_cpu_inputs import EVIDENCE, FIXTURE, MODEL, ROOT, make_experiment, read, sha256, verify_traces, window_data
from restore_edge_select.dynamic_cpu import StateGate
from restore_edge_select.offline_bounds import search
from study_dynamic_cpu_compact import replay
from study_dynamic_cpu_holdout import percent_interval


KEYS = ("trace", "window", "path", "rate", "placement", "spacing", "budget")
SOURCES = ("compact", "blocked", "io-margin", "entry", "entry-dense")


def candidates():
    result = []
    for name in SOURCES:
        source = EVIDENCE / f"dynamic-cpu-{name}.csv"
        grouped = {}
        with source.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                grouped.setdefault(tuple(row[k] for k in KEYS), {})[row["policy"]] = row
        for key, policies in grouped.items():
            row = policies["entry"]
            if row["oracle_exact"] == "0":
                width = float(row["oracle_upper_ms"]) - float(row["oracle_lower_ms"])
                result.append((width, source, key, policies))
    return sorted(result, key=lambda item: (-item[0], item[1].name, item[2]))


def refine(job, directory, budgets, resource_floor=False, dominance=False):
    width, source, key, archive = job
    values = dict(zip(KEYS, key, strict=True))
    fixture, model = read(FIXTURE), read(MODEL)["models"][0]
    _, _, _, trace = next(window_data(
        directory, values["trace"], (int(values["window"]),), fixture, model,
        ((values["placement"], F(values["spacing"])),), all_checkpoints="entry-dense" in source.name))
    exp = make_experiment(fixture, model, trace, values["path"], int(values["rate"]), int(values["budget"]))
    gate = StateGate(F(read(EVIDENCE / "dynamic-cpu-frozen.json")["thresholds"]["dynamic"]))
    runs = {policy: replay(exp, gate, policy) for policy in archive}
    for policy, actual in runs.items():
        assert abs(float(actual["response_mean_ms"]) - float(archive[policy]["zero_mean_ms"])) < 1e-7
    base, chosen = runs["local-cost"]["response_mean_ms"], runs["entry"]["response_mean_ms"]
    n = len(exp.requests)
    previous_lo = F(0)
    previous_hi = min(run["response_sum_ms"] for run in runs.values())
    for limit in budgets:
        started = perf_counter()
        diagnostics = {}
        lo, hi, plan, nodes = search(exp, runs.values(), limit, resource_floor=resource_floor,
                                      dominance=dominance, diagnostics=diagnostics)
        elapsed = perf_counter() - started
        assert previous_lo <= lo <= hi <= previous_hi
        actual = exp.run("fixed", boundaries=plan)
        assert actual["response_sum_ms"] == hi
        if limit == 32 and not (resource_floor or dominance):
            assert abs(float(lo / n) - float(archive["entry"]["oracle_lower_ms"])) < 1e-7
            assert abs(float(hi / n) - float(archive["entry"]["oracle_upper_ms"])) < 1e-7
        a, b, status = percent_interval(float(base), float(chosen), float(lo / n), float(hi / n))
        yield dict(
            source=source.name, **values, resource_floor=int(resource_floor), dominance=int(dominance),
            dominated=diagnostics["dominated"], distinct_states=diagnostics["distinct_states"],
            frontier=diagnostics["frontier"], node_budget=limit,
            nodes=nodes, search_seconds=elapsed,
            requests=n, baseline_ms=float(base), entry_ms=float(chosen),
            lower_ms=float(lo / n), upper_ms=float(hi / n), exact=int(lo == hi),
            original_width_ms=width, width_ms=float((hi - lo) / n),
            captured_lower_pct=a, captured_upper_pct=b, captured_status=status,
            feasible_boundaries=json.dumps(plan, separators=(",", ":")),
        )
        previous_lo, previous_hi = lo, hi
        if lo == hi:
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=ROOT / "data/raw/bailian")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--nodes", type=int, nargs="+", default=(32, 256, 2048))
    parser.add_argument("--resource-floor", action="store_true")
    parser.add_argument("--dominance", action="store_true")
    args = parser.parse_args()
    if args.top < 1 or args.nodes[0] != 32 or any(b <= a for a, b in zip(args.nodes, args.nodes[1:])):
        parser.error("Use a positive case count and increasing node budgets starting at 32")
    verify_traces(args.trace_dir)
    jobs = candidates()[:args.top]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as stream:
        writer = None
        for rank, job in enumerate(jobs, 1):
            for row in refine(job, args.trace_dir, args.nodes, resource_floor=args.resource_floor,
                              dominance=args.dominance):
                if writer is None:
                    writer = csv.DictWriter(stream, tuple(row), lineterminator="\n")
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                print(f"Case {rank}/{len(jobs)}: {row['nodes']} nodes, width {row['width_ms']:.3f} ms,"
                      f" exact={row['exact']}", flush=True)
    args.out.with_suffix(".json").write_text(json.dumps(dict(
        top=args.top, node_budgets=args.nodes, resource_floor=args.resource_floor, dominance=args.dominance,
        ranking="largest absolute offline interval first",
        selection=[dict(source=source.name, key=dict(zip(KEYS, key, strict=True)), width_ms=width)
                   for width, source, key, _ in jobs],
        source_sha256={p.relative_to(ROOT).as_posix(): sha256(p)
                       for p in (Path(__file__).resolve(), ROOT / "src/restore_edge_select/offline_bounds.py",
                                 *(EVIDENCE / f"dynamic-cpu-{name}.csv" for name in SOURCES))},
    ), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
