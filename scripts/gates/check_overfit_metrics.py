from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Enforce SFT overfit generation thresholds")
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--min-compact-exact", type=float, default=0.5)
    parser.add_argument("--min-chrf2", type=float, default=70.0)
    parser.add_argument("--max-replacement-rate", type=float, default=0.01)
    parser.add_argument("--max-length-stop-rate", type=float, default=0.05)
    args = parser.parse_args()

    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    metrics = payload["overall"]
    checks = {
        "compact_exact_match": (
            metrics["compact_exact_match"] >= args.min_compact_exact
        ),
        "mean_chrf2": metrics["mean_chrf2"] >= args.min_chrf2,
        "replacement_character_rate": (
            metrics["replacement_character_rate"] <= args.max_replacement_rate
        ),
        "length_truncation_rate": (
            metrics["length_truncation_rate"] <= args.max_length_stop_rate
        ),
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
