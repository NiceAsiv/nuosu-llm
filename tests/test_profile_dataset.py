from scripts.profile_dataset import token_length


class RecordingTokenizer:
    messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return [1, 2, 3]


def test_sft_token_length_combines_prompt_and_completion() -> None:
    tokenizer = RecordingTokenizer()
    row = {
        "prompt": [{"role": "user", "content": "问题 /no_think"}],
        "completion": [{"role": "assistant", "content": "答案"}],
    }

    assert token_length(tokenizer, row, "sft") == 3
    assert tokenizer.messages == [*row["prompt"], *row["completion"]]
