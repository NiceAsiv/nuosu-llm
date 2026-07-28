from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from nuosu_llm.benchmarks import (
    NUOSU_BENCH_DATASET_ID,
    NUOSU_BENCH_SPLIT,
    convert_nuosu_bench_record,
)


def main() -> None:
    from datasets import load_dataset

    parser = argparse.ArgumentParser(
        description="下载并转换完整 NuosuBench，用于基础模型比较。"
    )
    parser.add_argument("--output-dir", default="evaluation/nuosu_bench")
    parser.add_argument("--revision", help="可选的 Hugging Face 数据集 revision/commit")
    args = parser.parse_args()

    dataset = load_dataset(
        NUOSU_BENCH_DATASET_ID,
        split=NUOSU_BENCH_SPLIT,
        revision=args.revision,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test.jsonl"

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, record in enumerate(dataset):
            converted = convert_nuosu_bench_record(record, index)
            handle.write(json.dumps(converted, ensure_ascii=False) + "\n")

    manifest = {
        "dataset_id": NUOSU_BENCH_DATASET_ID,
        "source_split": NUOSU_BENCH_SPLIT,
        "revision": args.revision or "default",
        "rows": len(dataset),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "project_use": "base_model_comparison",
        "research_training_splits": (
            "../nuosu-corpus/data/processed/bootstrap_nuosu_bench"
        ),
        "output": str(output_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
