from scripts.model.smoke_test_base import assess_response


def test_assess_response_accepts_expected_clean_text():
    passed, reasons = assess_response("北京是中国的首都。", "北京")
    assert passed
    assert reasons == []


def test_assess_response_rejects_corruption_and_missing_answer():
    passed, reasons = assess_response("乱码\ufffd", "Paris")
    assert not passed
    assert reasons == [
        "unicode_replacement_character",
        "expected_text_missing",
    ]
