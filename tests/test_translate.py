import pytest

from nuosu_llm.translate import build_parser, validate_direction


def test_translate_cli_defaults_to_chinese_to_nuosu(monkeypatch):
    monkeypatch.delenv("NUOSU_BASE_MODEL", raising=False)
    monkeypatch.delenv("NUOSU_ADAPTER", raising=False)
    args = build_parser().parse_args([])
    assert args.source_lang == "zh"
    assert args.target_lang == "ii"
    assert args.max_new_tokens == 256


def test_translate_direction_rejects_non_nuosu_pair():
    validate_direction("zh", "ii")
    validate_direction("ii", "zh")
    with pytest.raises(ValueError, match="仅支持"):
        validate_direction("zh", "en")
