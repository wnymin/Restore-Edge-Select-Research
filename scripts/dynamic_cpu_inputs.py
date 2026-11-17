"""Shared, pinned inputs for the CPU intervention and holdout experiments."""

import csv
from fractions import Fraction as F
from hashlib import file_digest
from itertools import islice
import json

from study_bailian import Experiment, ROOT, build, read


EVIDENCE = ROOT / "docs/evidence"
FIXTURE = ROOT / "fixtures/phase1-hand-trace.json"
MODEL = ROOT / "configs/qwen3.5-4b.json"
GRID = EVIDENCE / "prefill-vulkan-4k.json"
IO = EVIDENCE / "load-staged-ssd-profile.json"
OVERLAP = ROOT / "configs/overlap-observed-upper.json"
ARCHIVE = EVIDENCE / "adaptive-trace-oracle.csv"
MANIFEST = ROOT / "data/bailian-manifest.json"
TRACES = ("qwen_coder_blksz_16.jsonl", "qwen_traceA_blksz_16.jsonl", "qwen_traceB_blksz_16.jsonl")


def sha256(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def verify_traces(directory):
    manifest = {item["name"]: item for item in read(MANIFEST)["files"]}
    for trace in TRACES:
        path = directory / trace
        entry = manifest[trace]
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Trace checksum mismatch: {path}")


def make_experiment(fixture, model, data, path, scale, budget):
    cost = dict(fixture["cost"], recompute=read(GRID))
    for tier, cfg in read(IO)["cost"].items():
        scale_path = tier == "ssd_to_device" or (path == "all" and tier == "ram_to_device")
        factor = scale if scale_path else 1
        cost[tier] = dict(startup_ms=str(F(cfg["startup_ms"]) * factor),
                          bytes_per_ms=str(F(cfg["bytes_per_ms"]) / factor))
    return Experiment(dict(data, cost=cost, overlap_slowdown=read(OVERLAP)), model, budget)


def window_data(directory, trace, windows, fixture, model, layouts, *, all_checkpoints=False):
    with (directory / trace).open(encoding="utf-8") as stream:
        offset = 0
        for window in sorted(windows):
            for _ in islice(stream, 352 * window - offset):
                pass
            rows = [json.loads(line, parse_float=str) for line in islice(stream, 352)]
            if len(rows) != 352:
                raise ValueError(f"Incomplete window: {trace} #{window}")
            offset = 352 * (window + 1)
            for placement, spacing in layouts:
                yield window, placement, spacing, build(rows, 256, fixture, model, spacing, placement, 3584,
                                                        all_checkpoints=all_checkpoints)


def archive_key(row):
    return (row["path"], int(row["io_scale"]), row["trace"], int(row["window"]),
            row["placement"], F(row["spacing"]), int(row["budget"]))


def archive_rows():
    with ARCHIVE.open(encoding="utf-8", newline="") as source:
        return {archive_key(row): row for row in csv.DictReader(source) if row["policy"] == "local-cost"}
