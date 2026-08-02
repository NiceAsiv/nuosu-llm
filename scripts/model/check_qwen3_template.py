from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EMPTY_THINKING_PREFIX = "<think>\n\n</think>\n\n"


def assess_rendering(
    *, prompt: str, completion: str, thinking_prompt: str, no_thinking_prompt: str
) -> list[str]:
    failures: list[str] = []
    if not prompt.endswith("<|im_start|>assistant\n"):
        failures.append("prompt_missing_assistant_generation_header")
    if not completion.startswith(EMPTY_THINKING_PREFIX):
        failures.append("completion_missing_empty_thinking_prefix")
    if thinking_prompt.endswith(EMPTY_THINKING_PREFIX):
        failures.append("thinking_mode_was_prefilled_as_non_thinking")
    if not no_thinking_prompt.endswith(EMPTY_THINKING_PREFIX):
        failures.append("no_thinking_mode_missing_empty_thinking_prefix")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Qwen3 thinking and contextual /no_think template semantics"
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from trl.data_utils import apply_chat_template

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=Path(args.tokenizer).is_dir(),
    )
    user = {"role": "user", "content": "请翻译：你好\n\n/no_think"}
    assistant = {"role": "assistant", "content": "ꀋꇊ"}
    rendered: dict[str, Any] = apply_chat_template(
        {"prompt": [user], "completion": [assistant]}, tokenizer
    )
    thinking_prompt = tokenizer.apply_chat_template(
        [user],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    no_thinking_prompt = tokenizer.apply_chat_template(
        [user],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    failures = assess_rendering(
        prompt=rendered["prompt"],
        completion=rendered["completion"],
        thinking_prompt=thinking_prompt,
        no_thinking_prompt=no_thinking_prompt,
    )
    template = tokenizer.chat_template or ""
    payload = {
        "tokenizer": args.tokenizer,
        "passed": not failures,
        "failures": failures,
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "prompt": rendered["prompt"],
        "completion": rendered["completion"],
        "thinking_prompt_suffix": thinking_prompt[-128:],
        "no_thinking_prompt_suffix": no_thinking_prompt[-128:],
    }
    output = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
