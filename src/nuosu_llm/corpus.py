from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_REPO_ID = "NiceAsiv/nuosu-corpus"
DEFAULT_REVISION = "v2026.08.04"
CORPUS_FILES = (
    "ready_cpt.jsonl",
    "ready_sft.jsonl",
    "validation_sft.jsonl",
    "research_test_eval.jsonl",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest_files(directory: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    declared = manifest.get("files")
    if not isinstance(declared, dict):
        raise ValueError("manifest.json 缺少 files 对象")

    verified: dict[str, dict[str, Any]] = {}
    for name in CORPUS_FILES:
        metadata = declared.get(name)
        if not isinstance(metadata, dict) or not metadata.get("sha256"):
            raise ValueError(f"manifest.json 缺少 {name} 的 SHA-256")
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"缺少训练文件: {path}")
        actual = sha256_file(path)
        expected = str(metadata["sha256"]).lower()
        if actual != expected:
            raise ValueError(f"{name} SHA-256 不匹配: expected={expected}, actual={actual}")
        verified[name] = {
            "bytes": path.stat().st_size,
            "records": metadata.get("records"),
            "sha256": actual,
        }
    return verified


def download_training_corpus(
    output_dir: str | Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
) -> dict[str, dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for filename in (
        "manifest.json",
        *CORPUS_FILES,
        "task_catalog.json",
        "CITATION.cff",
    ):
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            local_dir=root,
        )
    return verify_manifest_files(root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="下载并校验固定版本的 Nuosu CPT/SFT 训练语料"
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--output-dir",
        default="../nuosu-corpus/data/hf/nuosu-corpus",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    verified = download_training_corpus(
        args.output_dir,
        repo_id=args.repo_id,
        revision=args.revision,
    )
    print(json.dumps(verified, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
