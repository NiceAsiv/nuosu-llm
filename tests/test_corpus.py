import hashlib
import json
from pathlib import Path

import pytest

from nuosu_llm.corpus import verify_manifest_files


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_verify_manifest_files_accepts_pinned_training_files(tmp_path: Path) -> None:
    cpt = b'{"text":"cpt"}\n'
    sft = b'{"messages":[]}\n'
    validation = b'{"messages":[]}\n'
    research_test = b'{"messages":[],"reference":"answer"}\n'
    (tmp_path / "ready_cpt.jsonl").write_bytes(cpt)
    (tmp_path / "ready_sft.jsonl").write_bytes(sft)
    (tmp_path / "validation_sft.jsonl").write_bytes(validation)
    (tmp_path / "research_test_eval.jsonl").write_bytes(research_test)
    manifest = {
        "files": {
            "ready_cpt.jsonl": {"sha256": _sha256(cpt), "records": 1},
            "ready_sft.jsonl": {"sha256": _sha256(sft), "records": 1},
            "validation_sft.jsonl": {
                "sha256": _sha256(validation),
                "records": 1,
            },
            "research_test_eval.jsonl": {
                "sha256": _sha256(research_test),
                "records": 1,
            },
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = verify_manifest_files(tmp_path)

    assert result["ready_cpt.jsonl"]["records"] == 1
    assert result["ready_sft.jsonl"]["sha256"] == _sha256(sft)


def test_verify_manifest_files_rejects_hash_mismatch(tmp_path: Path) -> None:
    for name in (
        "ready_cpt.jsonl",
        "ready_sft.jsonl",
        "validation_sft.jsonl",
        "research_test_eval.jsonl",
    ):
        (tmp_path / name).write_text("changed", encoding="utf-8")
    manifest = {
        "files": {
            "ready_cpt.jsonl": {"sha256": "0" * 64},
            "ready_sft.jsonl": {"sha256": "0" * 64},
            "validation_sft.jsonl": {"sha256": "0" * 64},
            "research_test_eval.jsonl": {"sha256": "0" * 64},
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 不匹配"):
        verify_manifest_files(tmp_path)
