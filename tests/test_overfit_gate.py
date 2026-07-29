from scripts.gates.prepare_sft_overfit import project_record


def test_project_record_separates_training_target_from_evaluation_prompt():
    source = {
        "id": "example",
        "messages": [
            {"role": "user", "content": "翻译"},
            {"role": "assistant", "content": "ꀀꁌ"},
        ],
        "metadata": {"ignored": True},
    }

    train_record, eval_record = project_record(source)

    assert train_record == {
        "id": "example",
        "messages": source["messages"],
    }
    assert eval_record == {
        "id": "example",
        "messages": [{"role": "user", "content": "翻译"}],
        "reference": "ꀀꁌ",
        "split": "overfit_gate",
    }
