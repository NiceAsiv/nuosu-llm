from pathlib import Path

from scripts.capture_run_metadata import file_record


def test_file_record_captures_size_and_sha256(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_bytes(b"Nuosu\n")

    record = file_record(source)

    assert record["path"] == str(source.resolve())
    assert record["bytes"] == 6
    assert record["sha256"] == (
        "d61e335a660d5d097637091afbd28edd5360f87dea42ede2c3ace4a31c4b8b7c"
    )
