from __future__ import annotations

import argparse
import json
from collections import Counter
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
        "task": record.get("task"),
        "messages": messages,
    }
    eval_record = {
        "id": record.get("id"),
        "task": record.get("task"),
        "messages": messages[:-1],
        "reference": messages[-1].get("content", ""),
        "split": "overfit_gate",
    }
    return train_record, eval_record


def select_balanced_records(
    records: list[dict[str, Any]], *, limit: int, tasks: list[str]
) -> list[dict[str, Any]]:
    """Select a deterministic, near-uniform task sample in source order."""
    if limit < 1:
        raise ValueError("limit must be positive")
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("tasks must be a non-empty list of unique task names")

    base, remainder = divmod(limit, len(tasks))
    quotas = {
        task: base + (index < remainder) for index, task in enumerate(tasks)
    }
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for record in records:
        task = str(record.get("task") or "")
        if task not in quotas or counts[task] >= quotas[task]:
            continue
        selected.append(record)
        counts[task] += 1
        if len(selected) == limit:
            break
    missing = {task: quotas[task] - counts[task] for task in tasks if counts[task] < quotas[task]}
    if missing:
        raise ValueError(f"not enough rows for balanced tasks: {missing}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic train and generation views for the SFT overfit gate"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--eval-output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument(
        "--balanced-tasks",
        nargs="+",
        help="Select a deterministic near-uniform sample across these task names",
    )
    args = parser.parse_args()

    source_rows: list[dict[str, Any]] = []
    with args.input.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                source_rows.append(json.loads(line))
    if args.balanced_tasks:
        source_rows = select_balanced_records(
            source_rows, limit=args.limit, tasks=args.balanced_tasks
        )
    else:
        source_rows = source_rows[: args.limit]
    if len(source_rows) != args.limit:
        raise ValueError(f"requested {args.limit} rows, found {len(source_rows)}")

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        train_record, eval_record = project_record(source_row)
        train_rows.append(train_record)
        eval_rows.append(eval_record)

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
                "task_counts": dict(Counter(row.get("task") for row in train_rows)),
                "train_output": str(args.train_output),
                "eval_output": str(args.eval_output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
