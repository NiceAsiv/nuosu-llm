from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .unicode_utils import is_yi_syllable, normalize_text

LANGUAGE_NAMES = {"ii": "凉山规范彝文", "zh": "中文", "en": "英文"}
MT_PROMPTS = {
    ("zh", "ii"): "请将以下中文翻译为凉山规范彝文。只输出译文，不要解释。\n",
    ("ii", "zh"): "请将以下凉山规范彝文翻译为中文。只输出译文，不要解释。\n",
    ("en", "ii"): "请将以下英文翻译为凉山规范彝文。只输出译文，不要解释。\n",
    ("ii", "en"): "请将以下凉山规范彝文翻译为英文。只输出译文，不要解释。\n",
}

_DIRECTION_ALIASES = {
    "zh-to-ii": ("zh", "ii"),
    "zh-ii": ("zh", "ii"),
    "ii-to-zh": ("ii", "zh"),
    "ii-zh": ("ii", "zh"),
    "en-to-ii": ("en", "ii"),
    "en-ii": ("en", "ii"),
    "ii-to-en": ("ii", "en"),
    "ii-en": ("ii", "en"),
}
_QUOTED_TEXT = re.compile(r"“([^”]+)”|‘([^’]+)’|\"([^\"]+)\"|'([^']+)'")
_CJK = re.compile(r"[\u3400-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")
_SPACE = re.compile(r"\s+")
_META_EVALUATION_EXACT = frozenset(
    {
        "correct",
        "正确",
        "翻译正确",
        "是的，翻译正确",
        "正确，彝文和汉语对应准确",
    }
)
_META_EVALUATION_PREFIXES = ("错误，正确翻译应该是：", "错误，正确翻译应为：")


def contains_yi(text: str) -> bool:
    return any(is_yi_syllable(char) for char in text)


def build_mt_prompt(source_lang: str, target_lang: str, source_text: str) -> str:
    try:
        prefix = MT_PROMPTS[(source_lang, target_lang)]
    except KeyError as error:
        raise ValueError(f"不支持的翻译方向: {source_lang}->{target_lang}") from error
    return prefix + source_text.strip()


def _quoted_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for match in _QUOTED_TEXT.finditer(text):
        value = next(group for group in match.groups() if group is not None).strip()
        if value:
            candidates.append(value)
    return candidates


def _matches_language(text: str, language: str) -> bool:
    if language == "ii":
        return contains_yi(text)
    if language == "zh":
        return bool(_CJK.search(text)) and not contains_yi(text)
    if language == "en":
        return bool(_LATIN.search(text)) and not contains_yi(text)
    return False


def infer_direction(prompt: str, answer: str, metadata: dict[str, Any]) -> tuple[str, str] | None:
    raw_direction = str(metadata.get("direction") or "").strip().lower().replace("_", "-")
    if raw_direction in _DIRECTION_ALIASES:
        return _DIRECTION_ALIASES[raw_direction]

    normalized = normalize_text(prompt).casefold()
    target: str | None = None
    if any(
        marker in normalized
        for marker in (
            "翻译成凉山规范彝文",
            "翻译为凉山规范彝文",
            "翻译成彝文",
            "翻译为彝文",
            "用彝文表达",
            "的彝文是",
            "彝文翻译是什么",
            "in yi",
            "into yi",
            "yi language",
            "yi translation",
            "yi equivalent",
        )
    ):
        target = "ii"
    elif any(
        marker in normalized
        for marker in (
            "翻译成汉语",
            "翻译为汉语",
            "翻译成中文",
            "翻译为中文",
            "中文翻译是什么",
            "中文意思是什么",
            "在中文中怎么说",
            "用汉语怎么说",
            "用中文怎么说",
            "翻成中文",
            "在汉语中是什么意思",
            "意思用中文",
            "用汉语如何表达",
            "翻译一下这段彝文",
            "用汉语翻译",
            "转写为中文",
            "译为汉语",
            "用中文解释",
        )
    ):
        target = "zh"
    elif any(
        marker in normalized
        for marker in (
            "翻译成英文",
            "翻译为英文",
            "英文翻译是什么",
            "in english",
            "into english",
            "用英文解释",
            "用英语写出",
            "标准英语",
            "standard english",
            "english translation",
            "english equivalent",
        )
    ):
        target = "en"

    if target is None:
        if contains_yi(answer) and not contains_yi(prompt):
            target = "ii"
        elif contains_yi(prompt) and not contains_yi(answer):
            target = "zh" if _CJK.search(answer) else "en"
    if target == "ii":
        quoted = _quoted_candidates(prompt)
        source = (
            "en"
            if "英文" in normalized
            or "english" in normalized
            or any(_matches_language(candidate, "en") for candidate in quoted)
            else "zh"
        )
    elif target in {"zh", "en"} and contains_yi(prompt):
        source = "ii"
    else:
        return None
    direction = (source, target)
    return direction if direction in MT_PROMPTS else None


def extract_source(prompt: str, source_lang: str) -> str | None:
    candidates = [
        candidate
        for candidate in _quoted_candidates(prompt)
        if _matches_language(candidate, source_lang)
    ]
    if candidates:
        return max(candidates, key=len).strip()

    known_prefixes = tuple(MT_PROMPTS.values()) + (
        "请把下面的汉语翻译成凉山规范彝文：",
        "请把下面的凉山规范彝文翻译成汉语：",
        "请把下面的英文翻译成凉山规范彝文：",
        "请把下面的凉山规范彝文翻译成英文：",
    )
    for prefix in known_prefixes:
        if prompt.startswith(prefix):
            value = prompt[len(prefix) :].strip()
            return value or None
    for separator in ("\n", "：", ":"):
        if separator in prompt:
            value = prompt.rsplit(separator, 1)[-1].strip()
            if value and _matches_language(value, source_lang):
                return value
    return prompt.strip() if _matches_language(prompt, source_lang) else None


def extract_target(answer: str, target_lang: str, source_text: str) -> str | None:
    candidates = [
        candidate
        for candidate in _quoted_candidates(answer)
        if candidate != source_text and _matches_language(candidate, target_lang)
    ]
    if candidates:
        return max(candidates, key=len).strip()

    value = answer.strip()
    if target_lang == "ii" and contains_yi(value):
        if not _CJK.search(value):
            return value
        yi_positions = [index for index, char in enumerate(value) if is_yi_syllable(char)]
        yi_span = value[yi_positions[0] : yi_positions[-1] + 1].strip()
        if yi_span and not _CJK.search(yi_span):
            return yi_span
    prefixes = (
        "彝文的表达是",
        "彝文翻译为",
        "彝文翻译是",
        "彝文是",
        "中文意思是",
        "中文翻译为",
        "汉语翻译为",
        "英语翻译为",
        "英文翻译为",
    )
    for prefix in prefixes:
        if value.startswith(prefix):
            value = value[len(prefix) :].lstrip("：: ").strip()
            break
    value = value.strip("“”‘’\"'")
    return value if value and _matches_language(value, target_lang) else None


def _compact(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFC", text).casefold()
        if not char.isspace()
    )


def validate_pair(source: str, target: str, source_lang: str, target_lang: str) -> str | None:
    if not source or not target:
        return "empty_side"
    if "�" in source or "�" in target:
        return "replacement_character"
    if len(source) > 4096 or len(target) > 4096:
        return "too_long"
    if _compact(source) == _compact(target):
        return "identical_sides"
    if not _matches_language(source, source_lang):
        return "source_script_mismatch"
    if not _matches_language(target, target_lang):
        return "target_script_mismatch"
    return None


def quality_bucket(record: dict[str, Any], source: str, target: str) -> str:
    metadata = record.get("metadata") or {}
    source_id = str(metadata.get("source_id") or metadata.get("source_dataset") or "")
    quality = str(metadata.get("quality_tier") or metadata.get("quality") or "").casefold()
    if "yixueyanjiu-dict" in source_id or "dictionary" in quality:
        return "mt_translation_lexicon"
    if "published" in quality or quality.startswith("a_"):
        return "mt_translation_published"
    if len(_SPACE.sub("", source)) <= 4 and len(_SPACE.sub("", target)) <= 6:
        return "mt_translation_short"
    return "mt_translation_sentence"


def is_meta_evaluation_target(answer: str, record: dict[str, Any]) -> bool:
    """Detect benchmark answer-verdict text accidentally mixed into MT targets.

    NuosuBench bootstrap records contain both genuine translations and evaluator
    turns.  We therefore filter only verdict-shaped targets from that source,
    preserving literal translations such as the Chinese word “正确” elsewhere.
    Evaluation references are never passed through this filter.
    """

    metadata = record.get("metadata") or {}
    source_tag = " ".join(
        str(metadata.get(key) or "") for key in ("source_id", "source_dataset")
    )
    source_tag = f"{source_tag} {record.get('id') or ''}".casefold()
    if "nuosu-bench-bootstrap" not in source_tag:
        return False
    normalized = _SPACE.sub(" ", answer.strip()).casefold()
    return normalized in _META_EVALUATION_EXACT or any(
        normalized.startswith(prefix.casefold()) for prefix in _META_EVALUATION_PREFIXES
    )


def project_mt_record(
    record: dict[str, Any], *, evaluation: bool = False
) -> tuple[dict[str, Any] | None, str | None]:
    if (
        record.get("task") not in {None, "translation", "terminology_translation"}
        and not evaluation
    ):
        return None, "not_translation_task"
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return None, "invalid_messages"
    user_messages = [message for message in messages if message.get("role") == "user"]
    if not user_messages:
        return None, "missing_user"
    prompt = str(user_messages[-1].get("content") or "").strip()
    answer = str(record.get("reference") or "").strip()
    if not answer and messages[-1].get("role") == "assistant":
        answer = str(messages[-1].get("content") or "").strip()
    if not prompt or not answer:
        return None, "missing_prompt_or_answer"
    if not evaluation and is_meta_evaluation_target(answer, record):
        return None, "meta_evaluation_target"
    metadata = dict(record.get("metadata") or {})
    direction = infer_direction(prompt, answer, metadata)
    if direction is None:
        return None, "unknown_direction"
    source_lang, target_lang = direction
    source = extract_source(prompt, source_lang)
    if source is None:
        return None, "source_extraction_failed"
    target = extract_target(answer, target_lang, source)
    if target is None:
        return None, "target_extraction_failed"
    invalid_reason = validate_pair(source, target, source_lang, target_lang)
    if invalid_reason:
        return None, invalid_reason

    task = quality_bucket(record, source, target)
    fallback_id = hashlib.sha256(
        f"{source_lang}\0{target_lang}\0{source}\0{target}".encode()
    ).hexdigest()[:20]
    projected = {
        "id": str(record.get("id") or fallback_id),
        "task": task,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "source": source,
        "target": target,
        "messages": [
            {"role": "user", "content": build_mt_prompt(source_lang, target_lang, source)},
        ],
        "metadata": {
            **metadata,
            "mt_projection": "target_only_v1",
            "original_task": record.get("task"),
        },
    }
    if evaluation:
        projected["reference"] = target
        for key in ("benchmark", "dataset_id", "split", "source_row"):
            if key in record:
                projected[key] = record[key]
    else:
        projected["messages"].append({"role": "assistant", "content": target})
    return projected, None


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL 记录必须是对象")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
            count += 1
    return count, digest.hexdigest()


def prepare_split(
    input_path: str | Path, output_path: str | Path, *, evaluation: bool
) -> dict[str, Any]:
    projected_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    directions: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    seen: set[tuple[str, str, str, str]] = set()
    duplicates = 0
    for record in read_jsonl(input_path):
        projected, reason = project_mt_record(record, evaluation=evaluation)
        if projected is None:
            rejected[reason or "unknown"] += 1
            rejected_rows.append(
                {
                    "id": record.get("id"),
                    "reason": reason or "unknown",
                    "task": record.get("task"),
                    "messages": record.get("messages"),
                    "reference": record.get("reference"),
                    "metadata": record.get("metadata"),
                }
            )
            continue
        key = (
            projected["source_lang"],
            projected["target_lang"],
            _compact(projected["source"]),
            _compact(projected["target"]),
        )
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        projected_rows.append(projected)
        directions[f"{projected['source_lang']}->{projected['target_lang']}"] += 1
        tasks[projected["task"]] += 1
    count, sha256 = write_jsonl(output_path, projected_rows)
    rejected_path = Path(output_path).with_name(f"{Path(output_path).stem}.rejected.jsonl")
    rejected_count, rejected_sha256 = write_jsonl(rejected_path, rejected_rows)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "records": count,
        "sha256": sha256,
        "directions": dict(sorted(directions.items())),
        "tasks": dict(sorted(tasks.items())),
        "duplicates_removed": duplicates,
        "rejected": dict(sorted(rejected.items())),
        "rejected_audit": {
            "output": str(rejected_path),
            "records": rejected_count,
            "sha256": rejected_sha256,
        },
    }


def prepare_mt_dataset(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "nuosu_mt_dataset/1.0",
        "format": "target_only_translation",
        "prompt_templates": {
            f"{source}->{target}": prompt
            for (source, target), prompt in MT_PROMPTS.items()
        },
        "cleaning": {
            "drop_meta_evaluation_targets": True,
            "reason": "meta_evaluation_target",
            "source_scope": "nuosu-bench-bootstrap",
            "exact_targets": sorted(_META_EVALUATION_EXACT),
            "prefix_targets": list(_META_EVALUATION_PREFIXES),
        },
        "splits": {
            "train": prepare_split(train_path, target_dir / "train.jsonl", evaluation=False),
            "validation": prepare_split(
                validation_path, target_dir / "validation.jsonl", evaluation=False
            ),
            "test": prepare_split(test_path, target_dir / "test.jsonl", evaluation=True),
        },
    }
    (target_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
