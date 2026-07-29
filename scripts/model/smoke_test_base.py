from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROMPTS = (
    {"id": "zh-capital", "prompt": "中国的首都是", "expected": "北京"},
    {"id": "en-capital", "prompt": "The capital of France is", "expected": "Paris"},
    {"id": "arithmetic", "prompt": "1 + 1 =", "expected": "2"},
)


def assess_response(response: str, expected: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not response.strip():
        reasons.append("empty")
    if "\ufffd" in response:
        reasons.append("unicode_replacement_character")
    if expected.casefold() not in response.casefold():
        reasons.append("expected_text_missing")
    return not reasons, reasons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-fast generation gate for an untouched causal language model"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--minimum-passes", type=int, default=2)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": args.device},
    )
    model.eval()

    prompts = [sample["prompt"] for sample in PROMPTS]
    encoded = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_width = encoded["input_ids"].shape[-1]
    responses = tokenizer.batch_decode(
        generated[:, prompt_width:],
        skip_special_tokens=True,
    )
    results: list[dict[str, Any]] = []
    for sample, response in zip(PROMPTS, responses, strict=True):
        response = response.strip()
        passed, reasons = assess_response(response, sample["expected"])
        results.append(
            {
                **sample,
                "response": response,
                "passed": passed,
                "reasons": reasons,
            }
        )

    passed_count = sum(int(result["passed"]) for result in results)
    payload = {
        "model": args.model,
        "passed": passed_count >= args.minimum_passes,
        "passed_count": passed_count,
        "required_count": args.minimum_passes,
        "results": results,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
