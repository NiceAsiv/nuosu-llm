from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def split_text_losslessly(tokenizer: Any, text: str, max_tokens: int) -> list[str]:
    if not text:
        raise ValueError("CPT text 不能为空")
    if len(input_ids(tokenizer, text)) <= max_tokens:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        remaining = text[start:]
        remaining_tokens = len(input_ids(tokenizer, remaining))
        if remaining_tokens <= max_tokens:
            chunks.append(remaining)
            break

        estimated_chars = max(
            1,
            int(len(remaining) * max_tokens / remaining_tokens),
        )
        low = start + 1
        high = min(len(text), start + max(estimated_chars * 2, 2))
        while high < len(text) and len(input_ids(tokenizer, text[start:high])) <= max_tokens:
            low = high
            high = min(len(text), high + max(estimated_chars, 1))

        best = start + 1
        while low <= high:
            middle = (low + high) // 2
            if len(input_ids(tokenizer, text[start:middle])) <= max_tokens:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        chunk = text[start:best]
        if not chunk:
            raise RuntimeError("无法在 token 预算内生成非空 CPT 片段")
        chunks.append(chunk)
        start = best

    if "".join(chunks) != text:
        raise RuntimeError("CPT 分块未能无损重建原文")
    if any(len(input_ids(tokenizer, chunk)) > max_tokens for chunk in chunks):
        raise RuntimeError("CPT 分块超过 token 上限")
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a lossless, tokenizer-bounded derived CPT training view"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--source-revision")
    parser.add_argument("--tokenizer-revision")
    args = parser.parse_args()
    if args.max_tokens < 2:
        raise ValueError("max-tokens 必须至少为 2")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=Path(args.tokenizer).is_dir(),
        use_fast=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_records = 0
    output_chunks = 0
    split_records = 0
    source_characters = 0
    output_characters = 0
    total_tokens = 0
    maximum_tokens = 0

    with args.input.open("r", encoding="utf-8-sig") as source, args.output.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            text = str(record.get("text") or "")
            chunks = split_text_losslessly(tokenizer, text, args.max_tokens)
            source_records += 1
            source_characters += len(text)
            split_records += int(len(chunks) > 1)
            for chunk_index, chunk in enumerate(chunks):
                chunk_tokens = len(input_ids(tokenizer, chunk))
                derived = dict(record)
                derived["text"] = chunk
                derived["derived_chunk"] = {
                    "source_line": line_number,
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                }
                target.write(json.dumps(derived, ensure_ascii=False) + "\n")
                output_chunks += 1
                output_characters += len(chunk)
                total_tokens += chunk_tokens
                maximum_tokens = max(maximum_tokens, chunk_tokens)

    if source_characters != output_characters:
        raise RuntimeError("派生 CPT 文件字符总量与源文件不一致")
    manifest = {
        "schema": "nuosu_cpt_chunks/1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(args.input.resolve()),
            "revision": args.source_revision,
            "sha256": sha256_file(args.input),
            "records": source_records,
            "characters": source_characters,
        },
        "tokenizer": {
            "path": args.tokenizer,
            "revision": args.tokenizer_revision,
        },
        "transform": {
            "max_tokens": args.max_tokens,
            "overlap_tokens": 0,
            "lossless_character_reconstruction": True,
            "split_records": split_records,
        },
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
            "chunks": output_chunks,
            "characters": output_characters,
            "tokens_without_special_tokens": total_tokens,
            "maximum_chunk_tokens": maximum_tokens,
        },
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
