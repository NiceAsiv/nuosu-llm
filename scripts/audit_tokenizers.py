from __future__ import annotations

import argparse
import json
from pathlib import Path

from nuosu_llm.unicode_utils import count_yi_syllables, normalize_text


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="比较候选模型对规范彝文的分词效率")
    parser.add_argument("--input", required=True, help="UTF-8 文本，每行一条已核验句子")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", help="可选 JSON 结果路径")
    args = parser.parse_args()

    lines = [
        normalize_text(line)
        for line in Path(args.input).read_text(encoding="utf-8-sig").splitlines()
        if normalize_text(line)
    ]
    total_yi = sum(count_yi_syllables(line) for line in lines)
    if total_yi == 0:
        raise SystemExit("输入中没有检测到 U+A000–U+A48F 规范彝文字符")

    results = []
    for model_name in args.models:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        token_count = 0
        unknown_count = 0
        for line in lines:
            token_ids = tokenizer(line, add_special_tokens=False)["input_ids"]
            token_count += len(token_ids)
            if tokenizer.unk_token_id is not None:
                unknown_count += token_ids.count(tokenizer.unk_token_id)
        results.append(
            {
                "model": model_name,
                "sentences": len(lines),
                "yi_characters": total_yi,
                "tokens": token_count,
                "tokens_per_yi_character": round(token_count / total_yi, 4),
                "unknown_tokens": unknown_count,
            }
        )

    rendered = json.dumps(results, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

