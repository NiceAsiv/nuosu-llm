from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from nuosu_llm.unicode_utils import yi_ratio


def distributed_context(torch: Any) -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
    return world_size, rank, local_rank


def part_path(output_path: Path, rank: int) -> Path:
    return output_path.with_name(f"{output_path.name}.rank{rank}.jsonl")


def load_records(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            record["_source_index"] = source_index
            rows.append(record)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def generated_token_count(token_ids: list[int], eos_token_ids: set[int]) -> int:
    for index, token_id in enumerate(token_ids):
        if token_id in eos_token_ids:
            return index + 1
    return len(token_ids)


def generation_stop_token_ids(tokenizer: Any) -> set[int]:
    """Return the end markers used by Qwen Base and ChatML SFT."""
    eos_ids = tokenizer.eos_token_id
    stop_ids = (
        {int(eos_ids)}
        if isinstance(eos_ids, int)
        else {int(token_id) for token_id in eos_ids or []}
    )
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if (
        isinstance(im_end_id, int)
        and im_end_id >= 0
        and im_end_id != tokenizer.unk_token_id
    ):
        stop_ids.add(im_end_id)
    return stop_ids


def merge_parts(output_path: Path, world_size: int) -> int:
    rows: list[dict[str, Any]] = []
    for rank in range(world_size):
        with part_path(output_path, rank).open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    rows.sort(key=lambda row: row["source_index"])
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic batched evaluation with optional LoRA and multi-GPU sharding"
    )
    parser.add_argument("--model", required=True, help="Base model ID or local directory")
    parser.add_argument("--adapter", help="Optional PEFT/LoRA adapter directory")
    parser.add_argument("--tokenizer", help="Tokenizer source; defaults to adapter or model")
    parser.add_argument("--input", required=True, help='JSONL with {"id", "messages", "reference"}')
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-format",
        choices=("chat", "raw"),
        default="chat",
        help="Use the saved chat template or the final user content as a raw completion prompt",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the base model with the same NF4 quantization used by QLoRA training",
    )
    return parser


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    args = build_parser().parse_args()
    world_size, rank, local_rank = distributed_context(torch)
    started_at = time.perf_counter()
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    tokenizer_source = args.tokenizer or args.adapter or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "device_map": {"": local_rank},
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    all_records = load_records(Path(args.input), args.limit)
    rank_records = all_records[rank::world_size]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rank_output = part_path(output_path, rank)

    completed_ids: set[str] = set()
    if args.resume and rank_output.exists():
        with rank_output.open("r", encoding="utf-8") as handle:
            completed_ids = {
                str(json.loads(line).get("id")) for line in handle if line.strip()
            }
    mode = "a" if args.resume and rank_output.exists() else "w"
    pending = [
        record for record in rank_records if str(record.get("id")) not in completed_ids
    ]

    eos_token_ids = generation_stop_token_ids(tokenizer)
    generation_eos_token_id = sorted(eos_token_ids)

    with rank_output.open(mode, encoding="utf-8", newline="\n") as target:
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            if args.prompt_format == "chat":
                prompts = [
                    tokenizer.apply_chat_template(
                        record["messages"],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    for record in batch
                ]
            else:
                prompts = [
                    str(record["messages"][-1].get("content", "")) for record in batch
                ]
            inputs = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=args.max_input_tokens,
                return_tensors="pt",
            ).to(model.device)
            batch_started = time.perf_counter()
            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": generation_eos_token_id,
            }
            if args.do_sample:
                generation_kwargs.update(
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    **generation_kwargs,
                )
            elapsed = time.perf_counter() - batch_started
            prompt_width = inputs["input_ids"].shape[-1]
            new_token_rows = generated[:, prompt_width:].tolist()
            responses = tokenizer.batch_decode(
                new_token_rows,
                skip_special_tokens=True,
            )

            for record, response, token_ids in zip(
                batch, responses, new_token_rows, strict=True
            ):
                response = response.strip()
                target.write(
                    json.dumps(
                        {
                            "source_index": record["_source_index"],
                            "id": record.get("id"),
                            "benchmark": record.get("benchmark"),
                            "dataset_id": record.get("dataset_id"),
                            "split": record.get("split"),
                            "messages": record["messages"],
                            "reference": record.get("reference"),
                            "response": response,
                            "response_yi_ratio": round(yi_ratio(response), 6),
                            "generated_tokens": generated_token_count(
                                token_ids, eos_token_ids
                            ),
                            "stop_reason": (
                                "eos"
                                if any(
                                    token_id in eos_token_ids
                                    for token_id in token_ids
                                )
                                else "length"
                            ),
                            "batch_seconds_per_sample": round(
                                elapsed / len(batch), 6
                            ),
                            "human_score": None,
                            "human_notes": "",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            target.flush()
            completed = offset + len(batch)
            if completed % max(args.batch_size * 20, 1) == 0 or completed == len(
                pending
            ):
                print(
                    f"rank={rank} completed={completed}/{len(pending)} "
                    f"elapsed={time.perf_counter() - started_at:.1f}s",
                    flush=True,
                )

    if world_size > 1:
        torch.distributed.barrier()
    if rank == 0:
        merged_count = merge_parts(output_path, world_size)
        manifest = {
            "model": args.model,
            "adapter": args.adapter,
            "tokenizer": tokenizer_source,
            "input": args.input,
            "output": args.output,
            "records": merged_count,
            "world_size": world_size,
            "batch_size_per_gpu": args.batch_size,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else None,
            "top_p": args.top_p if args.do_sample else None,
            "seed": args.seed,
            "dtype": "bfloat16",
            "load_in_4bit": args.load_in_4bit,
            "prompt_format": args.prompt_format,
            "eos_token_ids": generation_eos_token_id,
        }
        output_path.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, ensure_ascii=False), flush=True)
    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
