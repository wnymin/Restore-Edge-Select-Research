"""Convert pure prompt-processing llama-bench JSON to a continuation grid."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from restore_edge_select.prefill_cost import sampled_cost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8-sig"))
    variable = {"n_depth", "n_prompt", "avg_ns", "stddev_ns", "avg_ts", "stddev_ts",
                "samples_ns", "samples_ts", "test_time"}
    metadata = [{k: v for k, v in row.items() if k not in variable} for row in rows]
    if not metadata or any(m != metadata[0] for m in metadata):
        parser.error("Need a grid from one model, build, and execution configuration")
    info = metadata[0]
    if info["n_gen"] != 0 or any(row["n_prompt"] <= 0 for row in rows):
        parser.error("Use only prompt processing (-n 0, with a positive -p)")
    if Path(info["model_filename"]).resolve() != args.model.resolve() or info["model_size"] <= 0:
        parser.error("Benchmark metadata does not match the model path")
    if any(not r["samples_ns"] or any(type(n) is not int or n <= 0 for n in r["samples_ns"])
           for r in rows):
        parser.error("Each shape needs positive integer nanosecond samples")
    with args.model.open("rb") as source:
        digest = file_digest(source, "sha256").hexdigest()
    config = dict(kind="sampled-prefill", provenance=dict(
        source="llama-bench", model_sha256=digest, **info),
        samples=[dict(cached_tokens=r["n_depth"], suffix_tokens=r["n_prompt"],
                      ms=[n / 1_000_000 for n in r["samples_ns"]]) for r in rows])
    sampled_cost(config)
    args.out.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
