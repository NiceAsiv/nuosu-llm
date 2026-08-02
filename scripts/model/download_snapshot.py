from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and ".cache" not in path.relative_to(root).parts
        and path.name not in {"VERIFIED.sha256", "snapshot_manifest.json"}
    )


def write_verification(
    root: Path, *, repo_id: str, revision: str
) -> dict[str, object]:
    files = snapshot_files(root)
    inventory = {
        path.relative_to(root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    }
    manifest = {
        "schema": "nuosu_model_snapshot/1.0",
        "repo_id": repo_id,
        "revision": revision,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "files": inventory,
    }
    manifest_path = root / "snapshot_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verification_files = [*files, manifest_path]
    (root / "VERIFIED.sha256").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
            for path in verification_files
        ),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a pinned Hugging Face model snapshot into an isolated directory"
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.output_dir,
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.model",
            "*.jinja",
            "*.safetensors",
        ],
        force_download=args.force_download,
    )
    manifest = write_verification(
        args.output_dir,
        repo_id=args.repo_id,
        revision=args.revision,
    )
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "revision": args.revision,
                "snapshot_path": snapshot_path,
                "verified_files": len(manifest["files"]),
                "verification_marker": str(args.output_dir / "VERIFIED.sha256"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
