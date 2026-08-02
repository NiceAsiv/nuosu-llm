from scripts.model.check_qwen3_template import (
    EMPTY_THINKING_PREFIX,
    assess_rendering,
)


def test_qwen3_template_gate_accepts_contextual_non_thinking_format() -> None:
    failures = assess_rendering(
        prompt="<|im_start|>user\nquestion /no_think<|im_end|>\n<|im_start|>assistant\n",
        completion=EMPTY_THINKING_PREFIX + "answer<|im_end|>\n",
        thinking_prompt="<|im_start|>assistant\n",
        no_thinking_prompt="<|im_start|>assistant\n" + EMPTY_THINKING_PREFIX,
    )

    assert failures == []
