from __future__ import annotations

from typing import Any

NUOSU_BENCH_DATASET_ID = "TianYeZ1214/NuosuBench"
NUOSU_BENCH_SPLIT = "test"


def build_benchmark_prompt(instruction: str, input_text: str = "") -> str:
    instruction = instruction.strip()
    input_text = input_text.strip()
    if not instruction:
        raise ValueError("NuosuBench 记录缺少 instruction")
    return f"{instruction}\n\n{input_text}" if input_text else instruction


def convert_nuosu_bench_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    instruction = record.get("instruction", "")
    input_text = record.get("input", "")
    reference = record.get("output", "")
    if not isinstance(instruction, str) or not isinstance(input_text, str):
        raise ValueError(f"NuosuBench 第 {index} 条 instruction/input 类型无效")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"NuosuBench 第 {index} 条缺少 output")
    return {
        "id": f"NuosuBench-test-{index:06d}",
        "benchmark": "NuosuBench",
        "dataset_id": NUOSU_BENCH_DATASET_ID,
        "split": NUOSU_BENCH_SPLIT,
        "source_row": index,
        "messages": [
            {
                "role": "user",
                "content": build_benchmark_prompt(instruction, input_text),
            }
        ],
        "reference": reference.strip(),
    }
