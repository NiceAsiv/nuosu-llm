from __future__ import annotations

import argparse
import json
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from nuosu_llm.unicode_utils import is_yi_syllable, normalize_text

_SPACE_AND_PUNCTUATION = re.compile(r"[\s\W_]+", re.UNICODE)


def compact_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).casefold()
    return _SPACE_AND_PUNCTUATION.sub("", normalized)


def infer_target_language(prompt: str) -> str:
    normalized = normalize_text(prompt)
    if "中文" in normalized or "汉语" in normalized:
        return "zh"
    if "彝文" in normalized or "彝语" in normalized:
        return "yi"
    return "unknown"


def levenshtein_distance(source: str, target: str) -> int:
    if len(source) < len(target):
        source, target = target, source
    previous = list(range(len(target) + 1))
    for source_index, source_char in enumerate(source, 1):
        current = [source_index]
        for target_index, target_char in enumerate(target, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[target_index] + 1,
                    previous[target_index - 1] + (source_char != target_char),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    return levenshtein_distance(reference, hypothesis) / max(len(reference), 1)


def ngram_counts(text: str, order: int) -> Counter[str]:
    return Counter(text[index : index + order] for index in range(len(text) - order + 1))


def chrf2(reference: str, hypothesis: str, max_order: int = 6) -> float:
    scores: list[float] = []
    beta_squared = 4.0
    for order in range(1, max_order + 1):
        reference_counts = ngram_counts(reference, order)
        hypothesis_counts = ngram_counts(hypothesis, order)
        if not reference_counts or not hypothesis_counts:
            continue
        overlap = sum((reference_counts & hypothesis_counts).values())
        precision = overlap / sum(hypothesis_counts.values())
        recall = overlap / sum(reference_counts.values())
        denominator = beta_squared * precision + recall
        scores.append(
            (1 + beta_squared) * precision * recall / denominator
            if denominator
            else 0.0
        )
    return statistics.mean(scores) * 100 if scores else 0.0


def yi_only(text: str) -> str:
    return "".join(char for char in text if is_yi_syllable(char))


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        reference = normalize_text(str(row.get("reference") or ""))
        response = normalize_text(str(row.get("response") or ""))
        compact_reference = compact_text(reference)
        compact_response = compact_text(response)
        prompt = str(row.get("messages", [{}])[-1].get("content", ""))
        target_language = infer_target_language(prompt)
        reference_yi = yi_only(reference)
        response_yi = yi_only(response)
        scored.append(
            {
                "target_language": target_language,
                "exact": response == reference,
                "compact_exact": compact_response == compact_reference,
                "reference_contained": bool(compact_reference)
                and compact_reference in compact_response,
                "chrf2": chrf2(compact_reference, compact_response),
                "cer": character_error_rate(compact_reference, compact_response),
                "empty": not response,
                "response_chars": len(response),
                "replacement_character": "\ufffd" in response,
                "length_truncated": row.get("stop_reason") == "length",
                "yi_exact": bool(reference_yi) and response_yi == reference_yi,
                "yi_reference_contained": bool(reference_yi)
                and reference_yi in response_yi,
                "generated_tokens": int(row.get("generated_tokens") or 0),
                "seconds": float(row.get("batch_seconds_per_sample") or 0.0),
            }
        )

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "records": len(items),
            "exact_match": round(mean([float(item["exact"]) for item in items]), 6),
            "compact_exact_match": round(
                mean([float(item["compact_exact"]) for item in items]), 6
            ),
            "reference_contained": round(
                mean([float(item["reference_contained"]) for item in items]), 6
            ),
            "mean_chrf2": round(mean([item["chrf2"] for item in items]), 4),
            "mean_cer": round(mean([item["cer"] for item in items]), 6),
            "empty_rate": round(mean([float(item["empty"]) for item in items]), 6),
            "replacement_character_rate": round(
                mean([float(item["replacement_character"]) for item in items]), 6
            ),
            "length_truncation_rate": round(
                mean([float(item["length_truncated"]) for item in items]), 6
            ),
            "mean_response_chars": round(
                mean([item["response_chars"] for item in items]), 3
            ),
            "yi_exact_match": round(
                mean([float(item["yi_exact"]) for item in items if item["target_language"] == "yi"]),
                6,
            ),
            "yi_reference_contained": round(
                mean(
                    [
                        float(item["yi_reference_contained"])
                        for item in items
                        if item["target_language"] == "yi"
                    ]
                ),
                6,
            ),
            "mean_generated_tokens": round(
                mean([item["generated_tokens"] for item in items]), 3
            ),
            "mean_seconds_per_sample": round(mean([item["seconds"] for item in items]), 6),
        }

    result = {"overall": summarize(scored), "by_target_language": {}}
    for language in ("yi", "zh", "unknown"):
        group = [item for item in scored if item["target_language"] == language]
        if group:
            result["by_target_language"][language] = summarize(group)
    return result


def render_markdown(label: str, metrics: dict[str, Any]) -> str:
    lines = [
        f"# Evaluation: {label}",
        "",
        "| Split | Records | Exact | chrF2 | CER | Empty | Length stop | Replacement char | Yi exact | Tokens | sec/sample |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    groups = [("overall", metrics["overall"])] + list(
        metrics["by_target_language"].items()
    )
    for name, values in groups:
        lines.append(
            f"| {name} | {values['records']} | {values['compact_exact_match']:.4f} | "
            f"{values['mean_chrf2']:.2f} | {values['mean_cer']:.3f} | "
            f"{values['empty_rate']:.4f} | {values['length_truncation_rate']:.4f} | "
            f"{values['replacement_character_rate']:.4f} | "
            f"{values['yi_exact_match']:.4f} | "
            f"{values['mean_generated_tokens']:.1f} | "
            f"{values['mean_seconds_per_sample']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score deterministic JSONL generations")
    parser.add_argument("--input", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8-sig") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    metrics = score_rows(rows)
    payload = {"label": args.label, **metrics}
    Path(args.output_json).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(args.output_markdown).write_text(
        render_markdown(args.label, metrics),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
