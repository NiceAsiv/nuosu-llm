from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def project_record(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("SFT record must contain at least user and assistant messages")
    if messages[-1].get("role") != "assistant":
        raise ValueError("SFT record must end with an assistant message")

    train_record = {
        "id": record.get("id"),
        "messages": messages,
    }
    eval_record = {
        "id": record.get("id"),
        "messages": messages[:-1],
        "reference": messages[-1].get("content", ""),
        "split": "overfit_gate",
    }
    return train_record, eval_record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic train and generation views for the SFT overfit gate"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--eval-output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args()

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    with args.input.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            train_record, eval_record = project_record(json.loads(line))
            train_rows.append(train_record)
            eval_rows.append(eval_record)
            if len(train_rows) >= args.limit:
                break
    if len(train_rows) != args.limit:
        raise ValueError(f"requested {args.limit} rows, found {len(train_rows)}")

    for path, rows in (
        (args.train_output, train_rows),
        (args.eval_output, eval_rows),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "source": str(args.input),
                "records": len(train_rows),
                "train_output": str(args.train_output),
                "eval_output": str(args.eval_output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
