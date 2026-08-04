import json

import pytest

from nuosu_llm.config import with_overrides
from nuosu_llm.training import (
    QWEN3_ASSISTANT_MASK_CHAT_TEMPLATE,
    ensure_assistant_mask_chat_template,
    initialize_distributed_runtime,
    load_stage_rows,
    require_verified_base_model,
    to_prompt_completion,
    training_sampling_strategy_name,
)


class FakeTokenizer:
    def __init__(self, chat_template=None):
        self.chat_template = chat_template


def test_distributed_runtime_is_noop_without_cuda():
    class Cuda:
        @staticmethod
        def is_available():
            return False

    class Torch:
        cuda = Cuda()

    initialize_distributed_runtime(Torch(), local_rank=0, world_size=3)


def test_training_sampling_strategy_is_resolved_without_trl_config_attribute():
    assert training_sampling_strategy_name(False) == "random"
    assert training_sampling_strategy_name(True) == "group_by_length"


def test_cpt_loader_ignores_heterogeneous_metadata(tmp_path):
    path = tmp_path / "cpt.jsonl"
    records = [
        {"text": "ꆈꌠ", "metadata": {"source_page": "cover"}},
        {"text": "ꀀꁌ", "metadata": {"source_page": 25}},
    ]
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    assert load_stage_rows(path, "cpt") == [
        {"text": "ꆈꌠ"},
        {"text": "ꀀꁌ"},
    ]


def test_sft_loader_keeps_only_messages(tmp_path):
    path = tmp_path / "sft.jsonl"
    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "ꀋꉬ"},
    ]
    path.write_text(
        json.dumps({"messages": messages, "metadata": {"page": 1}}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    assert load_stage_rows(path, "sft") == [{"messages": messages}]


def test_sft_loader_repeats_selected_tasks_without_dropping_others(tmp_path):
    path = tmp_path / "sft.jsonl"
    records = [
        {
            "task": "pronunciation_orthography",
            "messages": [
                {"role": "user", "content": "怎么读？"},
                {"role": "assistant", "content": "it"},
            ],
        },
        {
            "task": "translation",
            "messages": [
                {"role": "user", "content": "翻译"},
                {"role": "assistant", "content": "ꀋꉬ"},
            ],
        },
    ]
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    rows = load_stage_rows(
        path,
        "sft",
        repeat_by_task={"pronunciation_orthography": 3},
    )

    assert len(rows) == 4
    assert sum(row["messages"][-1]["content"] == "it" for row in rows) == 3
    assert sum(row["messages"][-1]["content"] == "ꀋꉬ" for row in rows) == 1


def test_sft_loader_rejects_invalid_task_repeat(tmp_path):
    path = tmp_path / "sft.jsonl"
    path.write_text(
        json.dumps(
            {
                "task": "single_turn_qa",
                "messages": [
                    {"role": "user", "content": "问题"},
                    {"role": "assistant", "content": "答案"},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="正整数"):
        load_stage_rows(path, "sft", repeat_by_task={"single_turn_qa": 0})


def test_prompt_completion_adds_contextual_no_think_without_mutating_source():
    messages = [
        {"role": "system", "content": "你是语言助手。"},
        {"role": "user", "content": "翻译成彝文：你好"},
        {"role": "assistant", "content": "ꀋꉬ"},
    ]

    converted = to_prompt_completion(messages, append_no_think=True)

    assert converted["prompt"][-1]["content"].endswith("/no_think")
    assert converted["completion"] == [{"role": "assistant", "content": "ꀋꉬ"}]
    assert messages[-2]["content"] == "翻译成彝文：你好"


def test_prompt_completion_requires_final_assistant():
    with pytest.raises(ValueError, match="最后一条消息为 assistant"):
        to_prompt_completion(
            [{"role": "user", "content": "只有问题"}],
            append_no_think=True,
        )


def test_loader_rejects_missing_stage_field(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"metadata": {}}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="缺少非空 text"):
        load_stage_rows(path, "cpt")


def test_installs_generation_spans_for_assistant_only_loss():
    tokenizer = FakeTokenizer("template without assistant spans")

    assert ensure_assistant_mask_chat_template(tokenizer, True)
    assert tokenizer.chat_template == QWEN3_ASSISTANT_MASK_CHAT_TEMPLATE
    assert "{% generation %}" in tokenizer.chat_template
    assert "{% endgeneration %}" in tokenizer.chat_template
    generation_span = tokenizer.chat_template.split("{% generation %}", 1)[1].split(
        "{% endgeneration %}", 1
    )[0]
    assert "<|endoftext|>" in generation_span
    assert "<|im_end|>" not in generation_span


def test_preserves_compatible_chat_template():
    template = "{% generation %}answer{% endgeneration %}"
    tokenizer = FakeTokenizer(template)

    assert not ensure_assistant_mask_chat_template(tokenizer, True)
    assert tokenizer.chat_template == template


def test_runtime_overrides_can_build_benchmarks_without_mutating_source():
    config = {
        "base_model": "Qwen/Qwen3-8B-Base",
        "output_dir": "outputs/original",
        "training": {"max_steps": -1, "resume_from_checkpoint": None},
    }

    updated = with_overrides(
        config,
        base_model="/models/verified",
        output_dir="outputs/benchmark",
        init_adapter="outputs/checkpoint-300",
        max_steps=30,
        resume_from_checkpoint="outputs/checkpoint-200",
    )

    assert config["output_dir"] == "outputs/original"
    assert config["base_model"] == "Qwen/Qwen3-8B-Base"
    assert config["training"]["max_steps"] == -1
    assert updated["output_dir"] == "outputs/benchmark"
    assert updated["base_model"] == "/models/verified"
    assert updated["init_adapter"] == "outputs/checkpoint-300"
    assert updated["training"]["max_steps"] == 30
    assert updated["training"]["resume_from_checkpoint"] == "outputs/checkpoint-200"


def test_requires_verified_local_base_model(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="VERIFIED.sha256"):
        require_verified_base_model({"base_model": str(model_dir)})

    marker = model_dir / "VERIFIED.sha256"
    marker.write_text("verified\n", encoding="utf-8")

    assert require_verified_base_model({"base_model": str(model_dir)}) == model_dir
