---
base_model: Qwen/Qwen3-8B-Base
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
language:
  - ii
  - zh
  - en
tags:
  - qwen3
  - lora
  - nuosu
  - yi
  - translation
  - academic-research
datasets:
  - TianYeZ1214/NuosuBench
  - nanxidajun/NuosuBburma-OCR-Evaluation-Set
---

# Qwen3-8B-Base Nuosu LoRA

This repository contains a research LoRA adapter for Standard Liangshan Yi
(Nuosu), based on `Qwen/Qwen3-8B-Base`. It is an adapter only: users must
separately obtain the original base model.

本仓库发布的是面向凉山规范彝文（诺苏语）的研究型 LoRA adapter，不包含
Qwen3-8B-Base 底模权重。

## Intended use / 适用范围

- Chinese–Nuosu and Nuosu–Chinese research translation;
- dictionary-style lookup and short-form generation;
- reproducible low-resource-language experiments.

The adapter is a research preview. It has not been validated for unrestricted
long-form dialogue, all Yi varieties, or high-stakes use. Native-speaker review
is required for semantic and orthographic claims.

该模型目前适合词典式查询、短句翻译和学术研究，不应宣称覆盖所有彝语方言，也不应在
未经母语者审核时用于高风险场景。

## Data / 数据

Training projections were built from:

- [NuosuBench](https://huggingface.co/datasets/TianYeZ1214/NuosuBench);
- [NuosuBburma OCR Evaluation Set](https://huggingface.co/datasets/nanxidajun/NuosuBburma-OCR-Evaluation-Set);
- [彝汉电子词典](https://www.yixueyanjiu.com/dict/);
- [yidir](https://github.com/isljsy/yidir), used as a Xide-pronunciation and
  character-mapping reference.

Reserved benchmark test projections were excluded from training.

## Evaluation / 评测

The formal held-out run contains 11,171 examples:

| Metric | Result |
|---|---:|
| Overall compact exact match | 17.84% |
| Overall chrF2 | 40.13 |
| Nuosu-target compact exact match | 3.38% |
| Output length truncation rate | 3.00% |

On a fixed 192-example validation gate, chrF2 improved from 10.09 for the
verified base model to 28.25 for this adapter, while length truncation dropped
from 64.58% to 0%. Overall metrics include Chinese, English, Nuosu and other
task formats, so the aggregate score must not be treated as Nuosu-only quality.

## Usage / 使用

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen3-8B-Base"
adapter_id = "NiceAsiv/Qwen3-8B-Base-Nuosu-LoRA"

tokenizer = AutoTokenizer.from_pretrained(adapter_id)
base = AutoModelForCausalLM.from_pretrained(
    base_id,
    torch_dtype="auto",
    device_map="auto",
)
model = PeftModel.from_pretrained(base, adapter_id)
```

The training and evaluation code is available in
[NiceAsiv/nuosu-llm](https://github.com/NiceAsiv/nuosu-llm).
