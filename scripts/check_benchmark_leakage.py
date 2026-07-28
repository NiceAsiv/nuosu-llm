from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from nuosu_llm.unicode_utils import normalize_text


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def strings_from_training_record(record: dict[str, Any]) -> Iterable[str]:
    text = record.get("text")
    if isinstance(text, str) and text.strip():
        yield text
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    yield content


def main() -> None:
    parser = argparse.ArgumentParser(
        description="检查训练 JSONL 与 NuosuBench 的精确文本重叠"
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--max-details", type=int, default=20)
    args = parser.parse_args()

    benchmark_hashes: dict[str, str] = {}
    with Path(args.benchmark).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for message in record.get("messages", []):
                content = message.get("content", "")
                if isinstance(content, str) and content.strip():
                    benchmark_hashes[fingerprint(content)] = f"{record.get('id')}:prompt"
            reference = record.get("reference", "")
            if isinstance(reference, str) and reference.strip():
                benchmark_hashes[fingerprint(reference)] = f"{record.get('id')}:reference"

    overlaps: list[dict[str, str]] = []
    training_strings = 0
    for train_path in args.train:
        with Path(train_path).open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                for text in strings_from_training_record(record):
                    training_strings += 1
                    matched = benchmark_hashes.get(fingerprint(text))
                    if matched:
                        overlaps.append(
                            {
                                "train_file": train_path,
                                "line": str(line_number),
                                "benchmark_record": matched,
                            }
                        )

    summary = {
        "benchmark_strings": len(benchmark_hashes),
        "training_strings": training_strings,
        "exact_overlaps": len(overlaps),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for overlap in overlaps[: args.max_details]:
        print(json.dumps(overlap, ensure_ascii=False))
    if overlaps:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

