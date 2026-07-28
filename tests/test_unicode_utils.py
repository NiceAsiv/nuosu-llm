from nuosu_llm.unicode_utils import (
    count_visible_characters,
    count_yi_syllables,
    normalize_text,
    yi_ratio,
)


def test_normalize_text() -> None:
    assert normalize_text("  A \n B  ") == "A B"


def test_yi_character_count() -> None:
    text = "\uA000\uA001中文"
    assert count_yi_syllables(text) == 2
    assert count_visible_characters(text) == 4
    assert yi_ratio(text) == 0.5

