import pytest

from nuosu_llm.benchmarks import (
    build_benchmark_prompt,
    convert_nuosu_bench_record,
)


def test_build_benchmark_prompt() -> None:
    assert build_benchmark_prompt("指令", "") == "指令"
    assert build_benchmark_prompt("指令", "输入") == "指令\n\n输入"


def test_convert_nuosu_bench_record() -> None:
    record = {"instruction": "翻译", "input": "\uA000\uA001", "output": "参考答案"}
    converted = convert_nuosu_bench_record(record, 7)
    assert converted["id"] == "NuosuBench-test-000007"
    assert converted["split"] == "test"
    assert converted["messages"][0]["content"] == "翻译\n\n\uA000\uA001"
    assert converted["reference"] == "参考答案"


def test_benchmark_record_requires_reference() -> None:
    with pytest.raises(ValueError):
        convert_nuosu_bench_record({"instruction": "翻译", "input": "", "output": ""}, 0)
