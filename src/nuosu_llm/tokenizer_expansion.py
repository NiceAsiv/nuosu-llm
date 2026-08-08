from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

YI_SYLLABLE_START = 0xA000
YI_SYLLABLE_END = 0xA48C
YI_RADICAL_START = 0xA490
YI_RADICAL_END = 0xA4C6


@dataclass(frozen=True)
class TokenizerExpansionPlan:
    requested_tokens: tuple[str, ...]
    added_tokens: tuple[str, ...]
    added_token_ids: tuple[int, ...]
    old_piece_ids: dict[int, tuple[int, ...]]
    before_vocab_size: int
    after_vocab_size: int
    coverage_before: dict[str, Any]

    def manifest(self) -> dict[str, Any]:
        token_digest = hashlib.sha256(
            "\n".join(self.added_tokens).encode("utf-8")
        ).hexdigest()
        return {
            "schema": "nuosu_tokenizer_expansion/1.0",
            "before_vocab_size": self.before_vocab_size,
            "after_vocab_size": self.after_vocab_size,
            "requested_tokens": len(self.requested_tokens),
            "added_tokens": len(self.added_tokens),
            "added_token_id_min": min(self.added_token_ids, default=None),
            "added_token_id_max": max(self.added_token_ids, default=None),
            "added_tokens_sha256": token_digest,
            "coverage_before": self.coverage_before,
            "initialization": "mean_of_original_subtoken_embeddings",
        }


def standard_yi_syllables() -> tuple[str, ...]:
    return tuple(chr(codepoint) for codepoint in range(YI_SYLLABLE_START, YI_SYLLABLE_END + 1))


def standard_yi_radicals() -> tuple[str, ...]:
    return tuple(chr(codepoint) for codepoint in range(YI_RADICAL_START, YI_RADICAL_END + 1))


def _deduplicate(tokens: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        normalized = unicodedata.normalize("NFC", str(token).strip())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def load_extra_tokens(path: str | Path | None) -> tuple[str, ...]:
    if not path:
        return ()
    token_path = Path(path)
    with token_path.open("r", encoding="utf-8-sig") as handle:
        return _deduplicate(
            line
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        )


def requested_tokens(config: dict[str, Any]) -> tuple[str, ...]:
    tokens: list[str] = []
    if config.get("add_standard_yi_syllables", True):
        tokens.extend(standard_yi_syllables())
    if config.get("add_yi_radicals", True):
        tokens.extend(standard_yi_radicals())
    tokens.extend(load_extra_tokens(config.get("extra_tokens_file")))
    return _deduplicate(tokens)


def _encode_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids = encoded.ids if hasattr(encoded, "ids") else encoded
    return tuple(int(token_id) for token_id in ids)


def token_coverage(tokenizer: Any, tokens: Iterable[str]) -> dict[str, Any]:
    lengths = [_encode_ids(tokenizer, token) for token in tokens]
    token_counts = [len(ids) for ids in lengths]
    single = sum(length == 1 for length in token_counts)
    return {
        "total": len(token_counts),
        "single_token": single,
        "single_token_rate": single / len(token_counts) if token_counts else 0.0,
        "mean_tokens": sum(token_counts) / len(token_counts) if token_counts else 0.0,
    }


def plan_tokenizer_expansion(
    tokenizer: Any, expansion_config: dict[str, Any]
) -> TokenizerExpansionPlan:
    tokens = requested_tokens(expansion_config)
    before_vocab_size = len(tokenizer)
    coverage_tokens = _deduplicate((*standard_yi_syllables(), *standard_yi_radicals()))
    coverage_encodings = {
        token: _encode_ids(tokenizer, token) for token in coverage_tokens
    }
    old_encodings = {
        token: coverage_encodings.get(token, _encode_ids(tokenizer, token))
        for token in tokens
    }
    missing = tuple(token for token, ids in old_encodings.items() if len(ids) != 1)
    added_count = int(tokenizer.add_tokens(list(missing)))
    if added_count != len(missing):
        raise ValueError(
            "tokenizer.add_tokens 未加入全部缺失 token: "
            f"expected={len(missing)} actual={added_count}"
        )
    added_ids = tuple(int(tokenizer.convert_tokens_to_ids(token)) for token in missing)
    if len(set(added_ids)) != len(added_ids):
        raise ValueError("新增 token ID 不唯一")
    old_piece_ids = {
        token_id: old_encodings[token]
        for token, token_id in zip(missing, added_ids, strict=True)
    }
    return TokenizerExpansionPlan(
        requested_tokens=tokens,
        added_tokens=missing,
        added_token_ids=added_ids,
        old_piece_ids=old_piece_ids,
        before_vocab_size=before_vocab_size,
        after_vocab_size=len(tokenizer),
        coverage_before={
            "standard_yi_syllables": {
                "total": len(standard_yi_syllables()),
                "single_token": sum(
                    len(coverage_encodings[token]) == 1
                    for token in standard_yi_syllables()
                ),
                "mean_tokens": sum(
                    len(coverage_encodings[token]) for token in standard_yi_syllables()
                )
                / len(standard_yi_syllables()),
            },
            "yi_radicals": {
                "total": len(standard_yi_radicals()),
                "single_token": sum(
                    len(coverage_encodings[token]) == 1
                    for token in standard_yi_radicals()
                ),
                "mean_tokens": sum(
                    len(coverage_encodings[token]) for token in standard_yi_radicals()
                )
                / len(standard_yi_radicals()),
            },
        },
    )


def resize_and_initialize_embeddings(
    model: Any,
    tokenizer: Any,
    plan: TokenizerExpansionPlan,
    *,
    pad_to_multiple_of: int = 64,
) -> None:
    if not plan.added_token_ids:
        return
    import torch

    old_input = model.get_input_embeddings().weight
    output_module = model.get_output_embeddings()
    old_output = output_module.weight if output_module is not None else None
    weights_were_tied = (
        old_output is not None and old_output.data_ptr() == old_input.data_ptr()
    )
    input_means: dict[int, Any] = {}
    output_means: dict[int, Any] = {}
    with torch.no_grad():
        for token_id in plan.added_token_ids:
            piece_ids = plan.old_piece_ids[token_id]
            if not piece_ids or any(piece_id >= old_input.shape[0] for piece_id in piece_ids):
                raise ValueError(f"新增 token {token_id} 的原始子词 ID 无法用于初始化")
            input_means[token_id] = old_input[list(piece_ids)].mean(dim=0).clone()
            if old_output is not None and not weights_were_tied:
                if any(piece_id >= old_output.shape[0] for piece_id in piece_ids):
                    raise ValueError(f"新增 token {token_id} 的输出层无法初始化")
                output_means[token_id] = old_output[list(piece_ids)].mean(dim=0).clone()
    try:
        model.resize_token_embeddings(
            len(tokenizer),
            pad_to_multiple_of=pad_to_multiple_of,
            mean_resizing=False,
        )
    except TypeError:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=pad_to_multiple_of)

    input_weight = model.get_input_embeddings().weight
    output_module = model.get_output_embeddings()
    output_weight = output_module.weight if output_module is not None else None
    weights_are_tied = (
        output_weight is not None and output_weight.data_ptr() == input_weight.data_ptr()
    )
    with torch.no_grad():
        for token_id in plan.added_token_ids:
            input_weight[token_id].copy_(input_means[token_id])
            if output_weight is not None and not weights_are_tied:
                if token_id not in output_means:
                    raise ValueError(f"新增 token {token_id} 的输出层无法初始化")
                output_weight[token_id].copy_(output_means[token_id])
    if getattr(model.config, "tie_word_embeddings", False):
        model.tie_weights()


def resize_model_to_tokenizer(
    model: Any, tokenizer: Any, *, pad_to_multiple_of: int = 64
) -> bool:
    """Resize an inference base before loading an adapter with added tokens."""
    current_size = int(model.get_input_embeddings().num_embeddings)
    if len(tokenizer) <= current_size:
        return False
    try:
        model.resize_token_embeddings(
            len(tokenizer),
            pad_to_multiple_of=pad_to_multiple_of,
            mean_resizing=False,
        )
    except TypeError:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=pad_to_multiple_of)
    return True


def write_expansion_manifest(path: str | Path, plan: TokenizerExpansionPlan) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(plan.manifest(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
