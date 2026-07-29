from nuosu_llm.chat import build_parser, generation_stop_token_ids


def test_chat_parser_supports_single_prompt():
    args = build_parser().parse_args(
        [
            "--model",
            "/verified/model",
            "--adapter",
            "/adapter",
            "--prompt",
            "测试",
        ]
    )
    assert args.prompt == "测试"
    assert args.temperature == 0.0
    assert args.max_new_tokens == 128


def test_chat_uses_qwen_and_chatml_stop_tokens():
    class Tokenizer:
        eos_token_id = 151643
        unk_token_id = None

        @staticmethod
        def convert_tokens_to_ids(token: str) -> int:
            return {"<|im_end|>": 151645, "<|endoftext|>": 151643}[token]

    assert generation_stop_token_ids(Tokenizer()) == [151643, 151645]
