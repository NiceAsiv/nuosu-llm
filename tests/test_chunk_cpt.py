from scripts.chunk_cpt import split_text_losslessly


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text)))}


def test_split_text_is_lossless_and_bounded() -> None:
    text = "彝语语料abcdefghi"

    chunks = split_text_losslessly(CharacterTokenizer(), text, max_tokens=4)

    assert "".join(chunks) == text
    assert all(1 <= len(chunk) <= 4 for chunk in chunks)
    assert len(chunks) == 4
