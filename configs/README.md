# Training configuration catalog / 训练配置目录

所有配置都需要一个已通过 `VERIFIED.sha256` 门禁的本地基础模型。运行时使用
`--base-model /absolute/path/to/verified/model` 覆盖示例 Model ID。

## 配置分级

| 类型 | 配置 | 用途 |
|---|---|---|
| 门禁 | `gates/sft_overfit_64.yaml` | 64 条确定性过拟合，不用于发布 |
| 模板 | `cpt_qwen3_8b_qlora.yaml` | 自建连续文本 CPT 起点 |
| 模板 | `sft_qwen3_8b_qlora.yaml` | 自建 messages SFT 起点 |
| 研究 | `cpt_qwen3_8b_ocr_gt_research.yaml` | OCR GT 单卡 CPT |
| 研究 | `sft_qwen3_8b_dictionary_research.yaml` | 词典单卡 SFT |
| 研究 | `sft_qwen3_8b_bootstrap.yaml` | NuosuBench 单卡 SFT |
| 三卡参考 | `*_3gpu.yaml` | 保守显存配置 |
| 三卡实验 | `*_3gpu_fast.yaml` | 经过特定机器画像的高吞吐配置 |

`fast` 配置不是跨机器默认值。换 GPU、PyTorch、attention backend 或数据长度分布后必须
重新运行 `profile_dataset.py` 和 `benchmark_throughput.sh`。

2026-07-29 以前生成的 adapter 基于损坏底模，均已作废；这不等于这些 YAML 参数已经通过
新底模验证。
