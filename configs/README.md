# Training configuration catalog / 训练配置目录

所有配置都需要一个已通过 `VERIFIED.sha256` 门禁的本地基础模型。运行时使用
`--base-model /absolute/path/to/verified/model` 覆盖示例 Model ID。

## 配置分级

| 类型 | 配置 | 用途 |
|---|---|---|
| 门禁 | `gates/sft_overfit_64.yaml` | 64 条确定性过拟合，不用于发布 |
| 配方 | `../recipes/qwen3-1.7b-full/` | 1.7B 全量 CPT + 全量 SFT 流水线 |
| 模板 | `cpt_qwen3_8b_qlora.yaml` | `ready_cpt.jsonl` 全量 CPT |
| 模板 | `sft_qwen3_8b_qlora.yaml` | `ready_sft.jsonl` 全量 SFT |

当前稳定配置固定使用 `NiceAsiv/nuosu-corpus@v2026.08.02`。先运行
`python scripts/download_training_corpus.py`，下载结果默认写入 sibling `nuosu-corpus` 仓库的
`data/hf/nuosu-corpus/`。旧 OCR-only、词典-only 和 NuosuBench 训练配置已经移除。

三卡和 `fast` 配置不是跨机器默认值，已移到
[`../experiments/three-gpu-24gb/configs/`](../experiments/three-gpu-24gb/configs/)。换 GPU、PyTorch、
attention backend 或数据长度分布后必须重新运行 `profile_dataset.py` 和
`benchmark_throughput.sh`。

2026-07-29 以前生成的 adapter 基于损坏底模，均已作废；这不等于这些 YAML 参数已经通过
新底模验证。
