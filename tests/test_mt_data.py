from __future__ import annotations

from nuosu_llm.mt_data import build_mt_prompt, project_mt_record


def test_projects_direct_dictionary_pair_to_target_only_mt():
    projected, reason = project_mt_record(
        {
            "id": "dict-1",
            "task": "translation",
            "metadata": {"source_id": "yixueyanjiu-dict", "direction": "zh-to-ii"},
            "messages": [
                {"role": "user", "content": "请把下面的汉语翻译成凉山规范彝文：学校"},
                {"role": "assistant", "content": "ꏃꃅꌠ"},
            ],
        }
    )

    assert reason is None
    assert projected is not None
    assert projected["source"] == "学校"
    assert projected["target"] == "ꏃꃅꌠ"
    assert projected["task"] == "mt_translation_lexicon"
    assert projected["messages"][-1] == {"role": "assistant", "content": "ꏃꃅꌠ"}


def test_projects_wrapped_benchmark_reference_without_answer_leakage():
    projected, reason = project_mt_record(
        {
            "id": "test-1",
            "messages": [{"role": "user", "content": "彝文“ꋿꁧꀉꑌꐯ”在中文中怎么说？"}],
            "reference": "彝文“ꋿꁧꀉꑌꐯ”的中文意思是“仪式常频繁”",
        },
        evaluation=True,
    )

    assert reason is None
    assert projected is not None
    assert projected["source_lang"] == "ii"
    assert projected["target_lang"] == "zh"
    assert projected["source"] == "ꋿꁧꀉꑌꐯ"
    assert projected["reference"] == "仪式常频繁"
    assert len(projected["messages"]) == 1


def test_projects_wrapped_yi_target():
    projected, reason = project_mt_record(
        {
            "id": "test-2",
            "messages": [{"role": "user", "content": "如何用彝文表达“蜀尸黑压压”？"}],
            "reference": "彝文的表达是“ꎰꁧꄹꇴꀕ”。",
        },
        evaluation=True,
    )

    assert reason is None
    assert projected is not None
    assert projected["source"] == "蜀尸黑压压"
    assert projected["reference"] == "ꎰꁧꄹꇴꀕ"


def test_projects_variant_chinese_translation_prompt():
    projected, reason = project_mt_record(
        {
            "id": "test-3",
            "messages": [{"role": "user", "content": "用汉语翻译以下彝文：ꌧꀑꋌꀉꁌ"}],
            "reference": "彝文“ꌧꀑꋌꀉꁌ”的中文意思是“先已牺牲启其头”",
        },
        evaluation=True,
    )

    assert reason is None
    assert projected is not None
    assert projected["source"] == "ꌧꀑꋌꀉꁌ"
    assert projected["reference"] == "先已牺牲启其头"


def test_strips_chinese_wrapper_from_unquoted_yi_target():
    projected, reason = project_mt_record(
        {
            "id": "test-4",
            "messages": [{"role": "user", "content": "如何用彝文表达“猴瘟一代堵而牢”？"}],
            "reference": "汉语词语“猴瘟一代堵而牢”的彝文是：ꑙꀉꋍꋏꆹ",
        },
        evaluation=True,
    )

    assert reason is None
    assert projected is not None
    assert projected["reference"] == "ꑙꀉꋍꋏꆹ"


def test_mt_prompt_requires_supported_direction():
    assert build_mt_prompt("zh", "ii", "你好").endswith("\n你好")


def test_drops_nuosubench_verdict_target_from_training():
    projected, reason = project_mt_record(
        {
            "id": "nuosu-bench-bootstrap-verdict-1",
            "task": "translation",
            "metadata": {"source_id": "nuosu-bench-bootstrap"},
            "messages": [
                {"role": "user", "content": "请将以下凉山规范彝文翻译为英文。\nꋑꎆ"},
                {"role": "assistant", "content": "Correct"},
            ],
        }
    )

    assert projected is None
    assert reason == "meta_evaluation_target"


def test_preserves_literal_correct_translation_from_other_source():
    projected, reason = project_mt_record(
        {
            "id": "dict-correct-1",
            "task": "translation",
            "metadata": {"source_id": "yixueyanjiu-dict", "direction": "ii-to-zh"},
            "messages": [
                {"role": "user", "content": "请将以下凉山规范彝文翻译为中文。\nꎃꊒ"},
                {"role": "assistant", "content": "正确"},
            ],
        }
    )

    assert reason is None
    assert projected is not None
    assert projected["target"] == "正确"


def test_recovers_corrected_translation_from_benchmark_verdict():
    projected, reason = project_mt_record(
        {
            "id": "nuosu-bench-bootstrap-correction-1",
            "task": "translation",
            "metadata": {"source_id": "nuosu-bench-bootstrap"},
            "messages": [
                {"role": "user", "content": "请将以下凉山规范彝文翻译为中文。\nꇉꇬꒉꌺ"},
                {"role": "assistant", "content": "错误，正确翻译应该是：洛果小河"},
            ],
        }
    )

    assert reason is None
    assert projected is not None
    assert projected["target"] == "洛果小河"
    assert projected["metadata"]["mt_cleaning"] == "meta_evaluation_correction_recovered"


def test_keeps_verdict_text_when_projecting_evaluation_reference():
    projected, reason = project_mt_record(
        {
            "id": "nuosu-bench-bootstrap-eval-1",
            "metadata": {"source_id": "nuosu-bench-bootstrap"},
            "messages": [{"role": "user", "content": "请将以下凉山规范彝文翻译为中文。\nꎃꊒ"}],
            "reference": "正确",
        },
        evaluation=True,
    )

    assert reason is None
    assert projected is not None
    assert projected["reference"] == "正确"
