from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_text(record: dict[str, Any]) -> str:
    reference = str(record.get("reference") or "").strip()
    if reference:
        return reference
    messages = record.get("messages") or []
    if messages and messages[-1].get("role") == "assistant":
        reference = str(messages[-1].get("content") or "").strip()
    if not reference:
        raise ValueError(f"{record.get('id', '<unknown>')}: missing reference")
    return reference


def bucket_records(
    input_path: Path,
    short_path: Path,
    long_path: Path,
    tokenizer: Any,
    threshold: int,
) -> dict[str, int]:
    if threshold < 1:
        raise ValueError("threshold must be positive")
    counts = {"total": 0, "short": 0, "long": 0, "max_reference_tokens": 0}
    short_path.parent.mkdir(parents=True, exist_ok=True)
    long_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        input_path.open("r", encoding="utf-8-sig") as source,
        short_path.open("w", encoding="utf-8", newline="\n") as short_target,
        long_path.open("w", encoding="utf-8", newline="\n") as long_target,
    ):
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            reference_tokens = len(
                tokenizer(reference_text(record), add_special_tokens=False)["input_ids"]
            )
            counts["total"] += 1
            counts["max_reference_tokens"] = max(
                counts["max_reference_tokens"], reference_tokens
            )
            if reference_tokens <= threshold:
                short_target.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["short"] += 1
            else:
                long_target.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["long"] += 1
    return counts


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(
        description="Split evaluation JSONL by reference-token length"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--threshold", type=int, default=1024)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    short_path = args.output_dir / f"reference-le-{args.threshold}.jsonl"
    long_path = args.output_dir / f"reference-gt-{args.threshold}.jsonl"
    counts = bucket_records(
        args.input, short_path, long_path, tokenizer, args.threshold
    )
    manifest = {
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "tokenizer": args.tokenizer,
        "threshold": args.threshold,
        "counts": counts,
        "files": {
            short_path.name: sha256_file(short_path),
            long_path.name: sha256_file(long_path),
        },
    }
    manifest_path = args.output_dir / "length-buckets.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
