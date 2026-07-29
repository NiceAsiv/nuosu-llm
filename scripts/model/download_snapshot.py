from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


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
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "revision": args.revision,
                "snapshot_path": snapshot_path,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
