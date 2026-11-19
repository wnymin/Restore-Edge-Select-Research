"""Audit pre-choice prediction separately from post-choice paired GPU outcomes."""

import argparse
from collections import Counter
import csv
from itertools import product
import json
from math import isfinite
from pathlib import Path
from random import Random
from statistics import median


DEVICE_TIMES = ("window_end_ms", "release_ms", "peer_d_ready_ms", "target_done_ms",
                "peer_c_done_ms", "peer_d_done_ms")


def extrema(samples):
    return [min(samples), max(samples)]


def calibration(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if int(r["rep"]) >= 0]
    distinct = [r for r in rows if r["mode"] == "distinct"]
    solo = [float(r["ssd_read_ms"]) + float(r["ssd_upload_ms"]) for r in rows if r["mode"] == "d-only"]
    blocked = [float(r["peer_d_ready_ms"]) - float(r["release_ms"])
               for r in distinct if int(r["blocked"])]
    return dict(isolated=median(solo), isolated_range=extrema(solo), blocked_range=extrema(blocked),
                jitter_range=extrema([float(r["window_end_ms"]) - float(r["nominal_window_ms"])
                                      for r in distinct]))


def marker_range(before, window, jitter):
    if before["marker_status"] == "done":
        return [0.0, 0.0]
    if before["marker_status"] != "pending":
        return None
    o0, o1 = before["origin_bracket_ns"]
    _, q1 = before["query_bracket_ns"]
    return [max(0, window + jitter[0] + (o0 - q1) / 1e6),
            max(0, window + jitter[1] + (o1 - q1) / 1e6)]


def score_delta(tight, work, remaining, isolated, after_grant):
    # Sum of B/C/D completion times, deep minus shallow. Idealized setup: 2ms
    # saved by each consumer downstream; shallow has work+2 before consuming D.
    deep_wait = max(0, after_grant - work) if tight else max(0, isolated - remaining - work)
    shallow_wait = max(0, isolated - remaining - work - 2)
    return -6 + deep_wait - shallow_wait


def predict(cell, before, cfg, fit):
    tight, work = cell["quota_mib"] == 80, cell["shallow_work_ms"]
    window, isolated = cfg["window_ms"], fit["isolated"]
    remaining = marker_range(before, window, fit["jitter_range"])
    flag = {"pending": window, "done": 0}.get(before["marker_status"])
    return remaining, {
        "isolated": [score_delta(tight, work, window, isolated, isolated)] * 2,
        "coarse_marker": None if flag is None else [score_delta(tight, work, flag, isolated, isolated)] * 2,
        # The delta is monotone in each argument separately (direction depends
        # on quota), so its extrema over this box occur at corners.
        "observable_interval": None if remaining is None else extrema([
            score_delta(tight, work, r, i, b) for r, i, b in
            product(remaining, fit["isolated_range"], fit["blocked_range"])])}


def winner(interval, tie):
    if interval is not None and interval[1] < -tie:
        return "deep"
    if interval is not None and interval[0] > tie:
        return "shallow"
    return None


def true_remaining(row):
    o0, o1 = row["before"]["origin_bracket_ns"]
    q1 = row["before"]["query_bracket_ns"][1]
    marker = row["after"]["window_end_ms"]
    return [max(0, marker + (o0 - q1) / 1e6), max(0, marker + (o1 - q1) / 1e6)]


def intersects(a, b):
    return None if a is None else a[0] <= b[1] and b[0] <= a[1]


def summarize_pairs(pairs, raw):
    summary = dict(n=len(pairs), truth=dict(sorted(Counter(x["truth"] or "tie" for x in pairs).items())),
                   flags=dict(sorted(Counter(r["before"]["marker_status"] for r in raw).items())),
                   missing=dict(active_compute=sum(r["before"]["active_compute"] is None for r in raw),
                                active_peer_transfer=sum(r["before"]["active_peer_transfer"] is None for r in raw)),
                   remaining_intersections=sum(x["marker_reference_intersects"] is True for x in pairs),
                   remaining_missing=sum(x["remaining_ms"] is None for x in pairs),
                   overhead_us={}, methods={})
    for label, key in [("query", "query_bracket_ns"), ("origin_sync_bracket", "origin_bracket_ns")]:
        times = [(r["before"][key][1] - r["before"][key][0]) / 1e3 for r in raw]
        summary["overhead_us"][label] = dict(median=median(times), maximum=max(times))
    for name in pairs[0]["predictions_ms"]:
        status = Counter()
        covered = 0
        for x in pairs:
            i = x["predictions_ms"][name]
            if i is not None and i[0] <= x["observed_delta_ms"] <= i[1]:
                covered += 1
            pick = x["picks"][name]
            if pick is None:
                status["missing" if i is None else "abstain"] += 1
            else:
                status["called"] += 1
                status["actual_tie" if x["truth"] is None else "right" if pick == x["truth"] else "wrong"] += 1
        summary["methods"][name] = dict(**{k: status[k] for k in
            ("called", "right", "wrong", "actual_tie", "abstain", "missing")}, delta_covered=covered)
    summary["observed_delta_ms"] = dict(range=extrema([x["observed_delta_ms"] for x in pairs]),
                                       median=median(x["observed_delta_ms"] for x in pairs))
    return summary


def audit(cfg, data, fit):
    if cfg != data["config"]:
        raise ValueError("Raw capture config does not match frozen config")
    cells = {c["id"]: c for c in cfg["cells"]}
    cases = list(product(cells, ("shallow", "deep")))
    rng = Random(cfg["seed"])
    expected = []
    for rep in range(-1, cfg["repeats"]):
        rng.shuffle(cases)
        expected.extend((rep, cell, arm) for cell, arm in cases)
    observations, outcomes = data["observations"], data["outcomes"]

    def key(row):
        return row["case_id"], row["arm"]

    truth_table = {key(r): r["after"] for r in outcomes}
    obs_keys = {key(r) for r in observations}
    if len(obs_keys) != len(observations) or len(truth_table) != len(outcomes) or obs_keys != truth_table.keys():
        raise ValueError("Capture tables must join one-to-one on (case_id, arm)")
    rows = [dict(r, after=truth_table[key(r)]) for r in observations]
    if [(r["rep"], r["cell"], r["arm"]) for r in rows] != expected:
        raise ValueError("Unexpected acquisition order, missing observations, or duplicate arms")
    by_case = {}
    for ordinal, row in enumerate(rows):
        cell, arm, rep = cells[row["cell"]], row["arm"], row["rep"]
        before, after = row["before"], row["after"]
        case = f"{cfg['protocol']}:{cell['id']}:{rep}"
        o0, o1 = before["origin_bracket_ns"]
        q0, q1 = before["query_bracket_ns"]
        size = 48 if arm == "deep" else 16
        union, cap = size + 64, cell["quota_mib"]
        blocked = union > cap
        peak = max(size + 32, 64) if blocked else union
        if (row["ordinal"] != ordinal or row["case_id"] != case or not 0 == o0 <= o1 <= q0 <= q1
            or before["marker_status"] != {0: "done", 600: "pending"}.get(before["query_code"], "unknown")
            or (after["rep"], after["arm"], after["mode"], after["nominal_window_ms"], after["quota_mib"],
                after["shallow_work_ms"], after["blocked"]) != (rep, arm, "distinct", cfg["window_ms"],
                    cap, cell["shallow_work_ms"], int(blocked))
            or (after["target_mib"], after["union_mib"], after["peak_mib"], after["copied_mib"]) !=
                (size, union, peak, union) or peak > cap
            or not all(isfinite(after[k]) and after[k] >= 0 for k in DEVICE_TIMES)
            or not after["window_end_ms"] <= after["release_ms"] <= after["target_done_ms"] <= after["peer_c_done_ms"]
            or after["peer_d_done_ms"] < max(after["peer_c_done_ms"], after["peer_d_ready_ms"])
            or (blocked and after["peer_d_ready_ms"] < after["release_ms"])):
            raise ValueError(f"Capture mismatch at {case} {arm}")
        if rep >= 0:
            by_case.setdefault(case, {})[arm] = row
    result = []
    for case, arms in sorted(by_case.items(), key=lambda item: (item[1]["shallow"]["cell"], item[1]["shallow"]["rep"])):
        s, d = arms["shallow"], arms["deep"]
        remaining, predictions = predict(cells[s["cell"]], s["before"], cfg, fit)
        reference = true_remaining(s)
        change = {k: d["after"][k] - s["after"][k] for k in DEVICE_TIMES}
        delta = sum(change[k] for k in DEVICE_TIMES[-3:])
        timing = {}
        for arm, row in arms.items():
            after = row["after"]
            release, ready, consume = (after[k] for k in ("release_ms", "peer_d_ready_ms", "peer_c_done_ms"))
            timing[arm] = dict(consume_after_release_ms=consume-release, ready_after_release_ms=ready-release,
                               exposed_ready_wait_ms=max(0, ready-consume))
        result.append(dict(case_id=case, cell=s["cell"], observed_delta_ms=delta, differences_ms=change,
                           timing=timing,
                           truth=winner([delta, delta], cfg["tie_ms"]), remaining_ms=remaining,
                           marker_reference_ms=reference, marker_reference_intersects=intersects(remaining, reference),
                           predictions_ms=predictions, picks={k: winner(v, cfg["tie_ms"]) for k, v in predictions.items()}))
    summary = {c: summarize_pairs([x for x in result if x["cell"] == c],
                                  [x for x in rows if x["rep"] >= 0 and x["cell"] == c]) for c in cells}
    return dict(protocol=cfg["protocol"], calibration=fit, cells=summary, pairs=result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/r02-gpu-observation-v1.json"))
    p.add_argument("--raw", type=Path, default=Path("docs/evidence/r02-gpu-observation-7941.json"))
    output = p.add_mutually_exclusive_group(required=True)
    output.add_argument("--write", type=Path)
    output.add_argument("--check", type=Path)
    a = p.parse_args()
    cfg = json.loads(a.config.read_bytes())
    data = json.loads(a.raw.read_bytes())
    report = audit(cfg, data, calibration(Path(cfg["calibration"])))
    if a.write:
        with a.write.open("x", encoding="utf-8") as f:
            f.write(json.dumps(report, indent=2) + "\n")
    elif report != json.loads(a.check.read_bytes()):
        p.exit(1, "Evidence differs from the recomputed audit\n")
    print(json.dumps(report["cells"], indent=2))


if __name__ == "__main__":
    main()
