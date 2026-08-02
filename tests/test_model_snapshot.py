from pathlib import Path

from scripts.model.download_snapshot import snapshot_files, write_verification


def test_write_verification_records_content_and_ignores_cache(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "weights.safetensors").write_bytes(b"weights")
    cache = tmp_path / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "download.lock").write_text("ignored", encoding="utf-8")

    manifest = write_verification(
        tmp_path,
        repo_id="Qwen/Qwen3-8B",
        revision="commit-sha",
    )

    assert manifest["repo_id"] == "Qwen/Qwen3-8B"
    assert set(manifest["files"]) == {"config.json", "weights.safetensors"}
    verification = (tmp_path / "VERIFIED.sha256").read_text(encoding="utf-8")
    assert "config.json" in verification
    assert "snapshot_manifest.json" in verification
    assert ".cache" not in verification
    assert snapshot_files(tmp_path) == [
        tmp_path / "config.json",
        tmp_path / "weights.safetensors",
    ]
