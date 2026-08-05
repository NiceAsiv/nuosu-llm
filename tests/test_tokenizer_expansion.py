from __future__ import annotations

from dataclasses import dataclass

from nuosu_llm.tokenizer_expansion import (
    plan_tokenizer_expansion,
    standard_yi_radicals,
    standard_yi_syllables,
)


@dataclass
class Encoding:
    ids: list[int]


class FakeTokenizer:
    def __init__(self) -> None:
        self.vocab = {"ꀀ": 1}
        self.next_id = 10

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = False) -> Encoding:
        del add_special_tokens
        if text in self.vocab:
            return Encoding([self.vocab[text]])
        return Encoding([2, 3, 4])

    def add_tokens(self, tokens: list[str]) -> int:
        for token in tokens:
            self.vocab[token] = self.next_id
            self.next_id += 1
        return len(tokens)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab[token]


def test_standard_yi_unicode_inventory_is_complete():
    assert len(standard_yi_syllables()) == 1165
    assert standard_yi_syllables()[0] == "ꀀ"
    assert standard_yi_syllables()[-1] == "ꒌ"
    assert len(standard_yi_radicals()) == 55


def test_expansion_adds_only_missing_tokens_and_keeps_old_piece_ids():
    tokenizer = FakeTokenizer()
    plan = plan_tokenizer_expansion(
        tokenizer,
        {
            "add_standard_yi_syllables": True,
            "add_yi_radicals": True,
        },
    )

    assert len(plan.requested_tokens) == 1220
    assert len(plan.added_tokens) == 1219
    assert "ꀀ" not in plan.added_tokens
    assert len(set(plan.added_token_ids)) == 1219
    assert set(plan.old_piece_ids.values()) == {(2, 3, 4)}
    assert plan.coverage_before["standard_yi_syllables"]["single_token"] == 1
    assert "added_token_ids" not in plan.manifest()
    assert plan.manifest()["added_token_id_min"] == 10
