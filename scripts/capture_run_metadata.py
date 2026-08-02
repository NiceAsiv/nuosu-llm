from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"只接受文件作为实验输入: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def command_output(command: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_record(cwd: Path) -> dict[str, Any]:
    revision = command_output(["git", "rev-parse", "HEAD"], cwd)
    branch = command_output(["git", "branch", "--show-current"], cwd)
    status = command_output(["git", "status", "--short"], cwd)
    return {
        "revision": revision,
        "branch": branch,
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
    }


def gpu_record(cwd: Path) -> list[dict[str, str]] | None:
    output = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        cwd,
    )
    if output is None:
        return None
    fields = ("index", "uuid", "name", "driver_version", "memory_total_mib")
    return [
        dict(zip(fields, (value.strip() for value in line.split(",")), strict=True))
        for line in output.splitlines()
        if line.strip()
    ]


def build_manifest(
    *, experiment_id: str, cwd: Path, inputs: list[Path], launch_command: str | None
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "launch_command": launch_command,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
        },
        "git": git_record(cwd),
        "gpus": gpu_record(cwd),
        "inputs": [file_record(path) for path in inputs],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture immutable provenance for a training or evaluation run"
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input", action="append", default=[], type=Path)
    parser.add_argument("--launch-command")
    args = parser.parse_args()

    manifest = build_manifest(
        experiment_id=args.experiment_id,
        cwd=Path.cwd(),
        inputs=args.input,
        launch_command=args.launch_command,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
