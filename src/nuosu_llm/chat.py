from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def generation_stop_token_ids(tokenizer: Any) -> list[int]:
    eos_ids = tokenizer.eos_token_id
    stop_ids = (
        {int(eos_ids)}
        if isinstance(eos_ids, int)
        else {int(token_id) for token_id in eos_ids or []}
    )
    for token in ("<|im_end|>", "<|endoftext|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if (
            isinstance(token_id, int)
            and token_id >= 0
            and token_id != tokenizer.unk_token_id
        ):
            stop_ids.add(token_id)
    if not stop_ids:
        raise ValueError("tokenizer 没有可用的 EOS token")
    return sorted(stop_ids)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interact with a verified local base model and optional LoRA adapter"
    )
    parser.add_argument("--model", required=True, help="已验证的本地基础模型目录")
    parser.add_argument("--adapter", help="可选的 PEFT/LoRA adapter 目录")
    parser.add_argument("--tokenizer", help="默认使用 adapter，其次使用 model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--system-prompt")
    parser.add_argument("--prompt", help="单次生成；省略时进入交互模式")
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser


def generate_response(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    messages: list[dict[str, str]],
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(
        rendered,
        truncation=True,
        max_length=max_input_tokens,
        return_tensors="pt",
    ).to(model.device)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": generation_stop_token_ids(tokenizer),
    }
    if temperature > 0:
        generation_kwargs.update(temperature=temperature, top_p=top_p)
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)
    prompt_tokens = inputs["input_ids"].shape[-1]
    response_tokens = generated[0, prompt_tokens:]
    return tokenizer.decode(response_tokens, skip_special_tokens=True).strip()


def main() -> None:
    args = build_parser().parse_args()
    model_path = Path(args.model).expanduser()
    if not model_path.is_dir() or not (model_path / "VERIFIED.sha256").is_file():
        raise SystemExit("--model 必须是含 VERIFIED.sha256 的已验证本地模型目录")
    if args.adapter and not (Path(args.adapter).expanduser() / "adapter_config.json").is_file():
        raise SystemExit("--adapter 缺少 adapter_config.json")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer_source = args.tokenizer or args.adapter or args.model
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
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    initial_messages = (
        [{"role": "system", "content": args.system_prompt}]
        if args.system_prompt
        else []
    )
    messages = list(initial_messages)

    def respond(user_prompt: str) -> str:
        messages.append({"role": "user", "content": user_prompt})
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            torch=torch,
            messages=messages,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        messages.append({"role": "assistant", "content": response})
        return response

    if args.prompt:
        print(respond(args.prompt))
        return

    print("Nuosu chat ready. 输入 /clear 清空上下文，/quit 退出。")
    while True:
        try:
            user_prompt = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_prompt:
            continue
        if user_prompt == "/quit":
            break
        if user_prompt == "/clear":
            messages = list(initial_messages)
            print("上下文已清空。")
            continue
        response = respond(user_prompt)
        print(f"模型> {response or '[空输出]'}")


if __name__ == "__main__":
    main()
