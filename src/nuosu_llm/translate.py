from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .chat import generate_response
from .mt_data import LANGUAGE_NAMES, MT_PROMPTS, build_mt_prompt
from .tokenizer_expansion import resize_model_to_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic Nuosu machine translation")
    parser.add_argument("--model", default=os.environ.get("NUOSU_BASE_MODEL"))
    parser.add_argument("--adapter", default=os.environ.get("NUOSU_ADAPTER"))
    parser.add_argument("--tokenizer", default=os.environ.get("NUOSU_TOKENIZER"))
    parser.add_argument("--source-lang", choices=("zh", "ii", "en"), default="zh")
    parser.add_argument("--target-lang", choices=("zh", "ii", "en"), default="ii")
    parser.add_argument("--text", help="单次翻译；省略时进入逐行交互模式")
    parser.add_argument("--device", default=os.environ.get("NUOSU_DEVICE", "cuda:0"))
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser


def validate_direction(source_lang: str, target_lang: str) -> None:
    if (source_lang, target_lang) not in MT_PROMPTS:
        raise ValueError(f"当前模型仅支持 ii↔zh 和 ii↔en，收到 {source_lang}->{target_lang}")


def load_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    if not args.model:
        raise SystemExit("请用 --model 或 NUOSU_BASE_MODEL 指定基础模型")
    if not args.adapter:
        raise SystemExit("请用 --adapter 或 NUOSU_ADAPTER 指定 MT adapter")
    adapter_path = Path(args.adapter).expanduser()
    if adapter_path.is_dir() and not (adapter_path / "adapter_config.json").is_file():
        raise SystemExit(f"adapter 缺少 adapter_config.json: {adapter_path}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer_source = args.tokenizer or args.adapter
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    model_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "device_map": {"": args.device},
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    resize_model_to_tokenizer(model, tokenizer)
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    return model, tokenizer, torch


def translate_once(
    *,
    text: str,
    source_lang: str,
    target_lang: str,
    model: Any,
    tokenizer: Any,
    torch: Any,
    max_input_tokens: int,
    max_new_tokens: int,
) -> str:
    validate_direction(source_lang, target_lang)
    prompt = build_mt_prompt(source_lang, target_lang, text)
    return generate_response(
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        messages=[{"role": "user", "content": prompt}],
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )


def main() -> None:
    args = build_parser().parse_args()
    validate_direction(args.source_lang, args.target_lang)
    model, tokenizer, torch = load_runtime(args)

    def run(text: str, source_lang: str, target_lang: str) -> str:
        return translate_once(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            model=model,
            tokenizer=tokenizer,
            torch=torch,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
        )

    if args.text:
        print(run(args.text, args.source_lang, args.target_lang))
        return

    source_lang, target_lang = args.source_lang, args.target_lang
    print("Nuosu MT ready. 输入 /swap 切换方向，/quit 退出。")
    while True:
        label = f"{LANGUAGE_NAMES[source_lang]}→{LANGUAGE_NAMES[target_lang]}"
        try:
            text = input(f"\n{label}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/swap":
            source_lang, target_lang = target_lang, source_lang
            print(f"已切换为 {LANGUAGE_NAMES[source_lang]}→{LANGUAGE_NAMES[target_lang]}")
            continue
        print(run(text, source_lang, target_lang) or "[空输出]")


if __name__ == "__main__":
    main()
