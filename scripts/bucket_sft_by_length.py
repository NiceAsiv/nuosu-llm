from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from nuosu_llm.config import load_config
from nuosu_llm.training import ensure_assistant_mask_chat_template


def message_token_length(tokenizer: Any, messages: list[dict[str, Any]]) -> int:
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    return len(token_ids)


def bucket_file(
    input_path: Path,
    short_path: Path,
    long_path: Path,
    tokenizer: Any,
    threshold: int,
) -> dict[str, int]:
    short_rows: list[str] = []
    long_rows: list[str] = []
    max_length = 0

    with input_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{input_path}:{line_number}: missing messages")
            length = message_token_length(tokenizer, messages)
            max_length = max(max_length, length)
            serialized = json.dumps(record, ensure_ascii=False) + "\n"
            if length <= threshold:
                short_rows.append(serialized)
            else:
                long_rows.append(serialized)

    short_path.write_text("".join(short_rows), encoding="utf-8")
    long_path.write_text("".join(long_rows), encoding="utf-8")
    return {
        "total": len(short_rows) + len(long_rows),
        "short": len(short_rows),
        "long": len(long_rows),
        "max_length": max_length,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split conversational SFT JSONL into token-length buckets"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--threshold", type=int, default=512)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    if config["stage"] != "sft":
        raise ValueError("length bucketing requires an SFT config")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], use_fast=True)
    ensure_assistant_mask_chat_template(
        tokenizer,
        bool(config["training"].get("assistant_only_loss", False)),
    )

    inputs = {"train": Path(config["train_file"])}
    eval_path = config.get("eval_file")
    if eval_path:
        inputs["validation"] = Path(eval_path)

    manifest = {
        "base_model": config["base_model"],
        "threshold": args.threshold,
        "splits": {},
    }
    for split, input_path in inputs.items():
        manifest["splits"][split] = bucket_file(
            input_path,
            output_dir / f"{split}_short.jsonl",
            output_dir / f"{split}_long.jsonl",
            tokenizer,
            args.threshold,
        )

    manifest_path = output_dir / "length_bucket_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
