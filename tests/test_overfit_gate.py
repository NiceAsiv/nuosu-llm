import pytest

from scripts.gates.prepare_sft_overfit import (
    project_record,
    select_balanced_directions,
    select_balanced_records,
)


def test_project_record_separates_training_target_from_evaluation_prompt():
    source = {
        "id": "example",
        "messages": [
            {"role": "user", "content": "翻译"},
            {"role": "assistant", "content": "ꀀꁌ"},
        ],
        "metadata": {"ignored": True},
    }

    train_record, eval_record = project_record(source)

    assert train_record == {
        "id": "example",
        "task": None,
        "messages": source["messages"],
    }
    assert eval_record == {
        "id": "example",
        "task": None,
        "messages": [{"role": "user", "content": "翻译"}],
        "reference": "ꀀꁌ",
        "split": "overfit_gate",
    }


def test_select_balanced_records_uses_deterministic_near_uniform_quotas():
    records = [
        {"id": f"a-{index}", "task": "a"} for index in range(4)
    ] + [
        {"id": f"b-{index}", "task": "b"} for index in range(4)
    ]

    selected = select_balanced_records(records, limit=5, tasks=["a", "b"])

    assert [row["id"] for row in selected] == ["a-0", "a-1", "a-2", "b-0", "b-1"]


def test_select_balanced_records_rejects_missing_task_quota():
    with pytest.raises(ValueError, match="not enough rows"):
        select_balanced_records(
            [{"id": "a-0", "task": "a"}], limit=2, tasks=["a", "b"]
        )


def test_select_balanced_directions_preserves_original_task():
    records = [
        {
            "id": f"zh-ii-{index}",
            "task": "mt_translation_sentence",
            "source_lang": "zh",
            "target_lang": "ii",
        }
        for index in range(3)
    ] + [
        {
            "id": f"ii-zh-{index}",
            "task": "mt_translation_lexicon",
            "source_lang": "ii",
            "target_lang": "zh",
        }
        for index in range(3)
    ]

    selected = select_balanced_directions(
        records, limit=4, directions=["zh->ii", "ii->zh"]
    )

    assert [row["id"] for row in selected] == [
        "zh-ii-0",
        "zh-ii-1",
        "ii-zh-0",
        "ii-zh-1",
    ]
    assert selected[0]["task"] == "mt_translation_sentence"
