import json

import pytest

from scripts.bucket_evaluation_by_reference_length import (
    bucket_records,
    reference_text,
)


class CharacterTokenizer:
    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": list(text)}


def test_bucket_records_preserves_rows_and_splits_by_reference_length(tmp_path):
    source = tmp_path / "input.jsonl"
    short = tmp_path / "short.jsonl"
    long = tmp_path / "long.jsonl"
    rows = [
        {"id": "short", "messages": [], "reference": "ab"},
        {
            "id": "long",
            "messages": [{"role": "assistant", "content": "abcd"}],
        },
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    counts = bucket_records(source, short, long, CharacterTokenizer(), threshold=3)

    assert counts == {"total": 2, "short": 1, "long": 1, "max_reference_tokens": 4}
    assert json.loads(short.read_text(encoding="utf-8")) == rows[0]
    assert json.loads(long.read_text(encoding="utf-8")) == rows[1]


def test_reference_text_rejects_missing_reference():
    with pytest.raises(ValueError, match="missing reference"):
        reference_text({"id": "missing", "messages": []})
