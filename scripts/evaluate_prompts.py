from __future__ import annotations

import argparse
import json
from pathlib import Path

from nuosu_llm.unicode_utils import yi_ratio


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description="运行固定提示集并输出待人工评分结果")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help='JSONL，每行包含 {"id", "messages"}')
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.input).open("r", encoding="utf-8-sig") as source, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record["messages"]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            target.write(
                json.dumps(
                    {
                        "id": record.get("id"),
                        "benchmark": record.get("benchmark"),
                        "dataset_id": record.get("dataset_id"),
                        "split": record.get("split"),
                        "messages": messages,
                        "reference": record.get("reference"),
                        "response": response,
                        "response_yi_ratio": round(yi_ratio(response), 6),
                        "human_score": None,
                        "human_notes": "",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
