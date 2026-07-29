from __future__ import annotations

from scripts.evaluate_prompts import generation_stop_token_ids, generated_token_count
from scripts.score_evaluation import (
    character_error_rate,
    chrf2,
    compact_text,
    infer_target_language,
    score_rows,
)


def test_generation_stops_at_all_qwen_end_markers():
    class Tokenizer:
        eos_token_id = 151643
        unk_token_id = None

        @staticmethod
        def convert_tokens_to_ids(token: str) -> int:
            assert token == "<|im_end|>"
            return 151645

    stop_ids = generation_stop_token_ids(Tokenizer())
    assert stop_ids == {151643, 151645}
    assert generated_token_count([7, 8, 151645, 9], stop_ids) == 3


def test_normalization_and_basic_metrics():
    assert compact_text(" 彝文：“ꀀ ꁌ” ") == "彝文ꀀꁌ"
    assert character_error_rate("abc", "abc") == 0.0
    assert character_error_rate("abc", "axc") == 1 / 3
    assert chrf2("ꀀꁌꂷ", "ꀀꁌꂷ") == 100.0


def test_target_language_inference():
    assert infer_target_language("这句话的中文意思是什么？") == "zh"
    assert infer_target_language("如何用彝文表达这句话？") == "yi"
    assert (
        infer_target_language(
            "请把下面的汉语翻译成凉山规范彝文：测试",
            "ꀀꁌ",
        )
        == "yi"
    )
    assert infer_target_language("回答问题") == "unknown"


def test_score_rows_reports_language_groups():
    rows = [
        {
            "messages": [{"role": "user", "content": "如何用彝文表达“测试”？"}],
            "reference": "ꀀꁌ",
            "response": "ꀀꁌ",
            "generated_tokens": 3,
            "batch_seconds_per_sample": 0.1,
            "stop_reason": "eos",
        },
        {
            "messages": [{"role": "user", "content": "它的中文意思是什么？"}],
            "reference": "测试",
            "response": "测试",
            "generated_tokens": 2,
            "batch_seconds_per_sample": 0.2,
            "stop_reason": "length",
        },
    ]

    metrics = score_rows(rows)

    assert metrics["overall"]["records"] == 2
    assert metrics["overall"]["compact_exact_match"] == 1.0
    assert metrics["overall"]["length_truncation_rate"] == 0.5
    assert metrics["overall"]["replacement_character_rate"] == 0.0
    assert metrics["by_target_language"]["yi"]["yi_exact_match"] == 1.0
    assert metrics["by_target_language"]["zh"]["records"] == 1
