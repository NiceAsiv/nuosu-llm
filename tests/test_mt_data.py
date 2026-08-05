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


def test_mt_prompt_requires_supported_direction():
    assert build_mt_prompt("zh", "ii", "你好").endswith("\n你好")
