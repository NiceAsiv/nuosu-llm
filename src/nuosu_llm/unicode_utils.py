from __future__ import annotations

import re
import unicodedata

YI_SYLLABLE_START = 0xA000
YI_SYLLABLE_END = 0xA48F
YI_RADICAL_START = 0xA490
YI_RADICAL_END = 0xA4CF

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize Unicode and collapse repeated whitespace."""
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def is_yi_syllable(char: str) -> bool:
    return len(char) == 1 and YI_SYLLABLE_START <= ord(char) <= YI_SYLLABLE_END


def is_yi_radical(char: str) -> bool:
    return len(char) == 1 and YI_RADICAL_START <= ord(char) <= YI_RADICAL_END


def count_yi_syllables(text: str) -> int:
    return sum(is_yi_syllable(char) for char in text)


def count_visible_characters(text: str) -> int:
    return sum(not char.isspace() for char in text)


def yi_ratio(text: str) -> float:
    visible = count_visible_characters(text)
    return count_yi_syllables(text) / visible if visible else 0.0

