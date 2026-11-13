"""Run a Phase 1 trace and print JSON results."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from restore_edge_select.simulator import Experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=ROOT / "fixtures/phase1-hand-trace.json")
    parser.add_argument("--model", type=Path, default=ROOT / "configs/qwen3.5-4b.json")
    parser.add_argument("--prefetch-mib", type=int, nargs="+")
    parser.add_argument("--policy", nargs="+",
                        choices=["recompute", "longest", "threshold", "local-cost", "capacity", "offline"],
                        default=["local-cost", "capacity", "offline"])
    parser.add_argument("--threshold-tokens", type=int, default=576)
    parser.add_argument("--max-combinations", type=int, default=10000)
    args = parser.parse_args()
    try:
        trace = json.loads(args.trace.read_text(encoding="utf-8"))
        models = json.loads(args.model.read_text(encoding="utf-8"))["models"]
        model = next((m for m in models if m["model_id"] == trace["model"]
                      and f"/{trace['revision']}/" in m["config_url"]), None)
        if model is None:
            raise ValueError("No model matches the trace model ID and revision")
        results = []
        for budget in args.prefetch_mib or trace["memory"]["prefetch_mib_cases"]:
            exp = Experiment(trace, model, budget)
            for policy in args.policy:
                run = (exp.offline(args.max_combinations) if policy == "offline" else
                       exp.run(policy, threshold_tokens=args.threshold_tokens))
                results.append({"prefetch_mib": budget, "policy": policy, **run})
        print(json.dumps(results, indent=2, default=lambda x: float(x) if isinstance(x, Fraction) else x))
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
