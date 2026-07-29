import json

import pytest

from nuosu_llm.config import with_overrides
from nuosu_llm.training import (
    QWEN3_ASSISTANT_MASK_CHAT_TEMPLATE,
    ensure_assistant_mask_chat_template,
    load_stage_rows,
    require_verified_base_model,
)


class FakeTokenizer:
    def __init__(self, chat_template=None):
        self.chat_template = chat_template


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
