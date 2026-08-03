from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nuosu_llm.config import load_config
from nuosu_llm.training import (
    ensure_assistant_mask_chat_template,
    load_stage_rows,
)


def percentile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, math.ceil(fraction * len(sorted_values)) - 1)
    return sorted_values[max(index, 0)]


def grouped_padding_tokens(lengths: list[int], batch_size: int) -> int:
    ordered = sorted(lengths)
    padded = 0
    for offset in range(0, len(ordered), batch_size):
        batch = ordered[offset : offset + batch_size]
        padded += max(batch) * len(batch)
    return padded


def token_length(
    tokenizer: Any,
    row: dict[str, Any],
    stage: str,
) -> int:
    if stage == "sft":
        messages = row.get("messages")
        if messages is None:
            messages = [*row["prompt"], *row["completion"]]
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        if isinstance(token_ids, Mapping):
            token_ids = token_ids["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
    else:
        token_ids = tokenizer(row["text"], add_special_tokens=True)["input_ids"]
    return len(token_ids)


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(
        description="Profile token lengths and batching efficiency for a training config"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--base-model", help="Local tokenizer/model path override")
    args = parser.parse_args()

    config = load_config(args.config)
    stage = config["stage"]
    training = config["training"]
    prompt_completion = bool(training.get("prompt_completion", False))
    append_no_think = bool(training.get("append_no_think", False))
    rows = load_stage_rows(
        config["train_file"],
        stage,
        prompt_completion=prompt_completion,
        append_no_think=append_no_think,
    )
    if args.limit:
        rows = rows[: args.limit]

    tokenizer_source = args.base_model or config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if stage == "sft":
        ensure_assistant_mask_chat_template(
            tokenizer,
            bool(config["training"].get("assistant_only_loss", False)),
        )

    max_length = int(training.get("max_length", 2048))
    raw_lengths = [token_length(tokenizer, row, stage) for row in rows]
    lengths = [min(length, max_length) for length in raw_lengths]
    ordered = sorted(lengths)
    total_tokens = sum(lengths)
    raw_total_tokens = sum(raw_lengths)
    discarded_tokens = raw_total_tokens - total_tokens
    padded_tokens = grouped_padding_tokens(lengths, args.batch_size)
    packed_sequences = math.ceil(total_tokens / max_length)

    result = {
        "config": str(Path(args.config)),
        "tokenizer": tokenizer_source,
        "prompt_completion": prompt_completion,
        "append_no_think": append_no_think,
        "samples": len(lengths),
        "max_length": max_length,
        "truncated_samples": sum(length > max_length for length in raw_lengths),
        "raw_max_length": max(raw_lengths),
        "tokens": {
            "raw_total": raw_total_tokens,
            "total": total_tokens,
            "discarded_by_truncation": discarded_tokens,
            "retained_ratio": round(total_tokens / raw_total_tokens, 6),
            "mean": round(total_tokens / len(lengths), 2),
            "p50": percentile(ordered, 0.50),
            "p90": percentile(ordered, 0.90),
            "p95": percentile(ordered, 0.95),
            "p99": percentile(ordered, 0.99),
            "max": max(ordered),
        },
        "length_grouped_batching": {
            "batch_size": args.batch_size,
            "padding_efficiency": round(total_tokens / padded_tokens, 4),
        },
        "theoretical_packing": {
            "fixed_length_sequences": packed_sequences,
            "samples_per_sequence": round(len(lengths) / packed_sequences, 2),
            "compression_ratio": round(len(lengths) / packed_sequences, 2),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
