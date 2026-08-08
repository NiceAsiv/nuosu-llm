# Command catalog / 命令目录

## 推荐入口

| 阶段 | 命令 |
|---|---|
| 模型下载与校验 | 见 `scripts/model/README.md` |
| 固定版本语料下载 | `python scripts/download_training_corpus.py` |
| 论文实验环境 | `bash scripts/setup_research_env.sh` |
| 64 条过拟合门禁 | `bash scripts/gates/run_sft_overfit_64.sh /verified/model` |
| 数据画像 | `python scripts/profile_dataset.py --config ...` |
| 单卡交互试用 | `nuosu-chat --model /verified/model --adapter /adapter` |
| 专用中彝翻译 | `bash scripts/translate_1_7b.sh` 或 `nuosu-translate ...` |
| 训练 | `python scripts/train.py --config ... --base-model ...` |
| 多卡训练 | `torchrun --nproc_per_node=N scripts/train.py ...` |
| 多卡评测 | `torchrun --nproc_per_node=N scripts/evaluate_prompts.py ...` |
| 自动评分 | `python scripts/score_evaluation.py ...` |
| 1.7B post-trained 全流程 | `NUM_GPUS=3 bash recipes/qwen3-1.7b-mt-post/run.sh /verified/Qwen3-1.7B` |
| 1.7B Base+CPT 负面对照 | `NUM_GPUS=3 bash recipes/qwen3-1.7b-mt/run.sh` |
| 合并 adapter | `python scripts/merge_adapter.py ...` |

## 目录职责

- `model/`：下载、实际 SHA-256 校验和未训练底模冒烟；
- `gates/`：小样本过拟合和机器可判定的停止条件；
- 根目录 Python 脚本：当前训练、评测、语料画像与泄漏审计工具；
- `prepare_mt_dataset.py`：把统一任务语料投影为纯译文 MT train/validation/test，并输出拒绝审计；
- `translate_1_7b.sh`：自动使用服务器默认模型与 adapter 的简短翻译入口；
- `benchmark_throughput.sh`：GPU 空闲时运行的独立吞吐基准。

特定服务器编排位于 `experiments/`，事故恢复位于 `patches/`。它们不是稳定命令目录的一部分。

每个命令均支持 `--help`。自动化脚本必须在 README 中说明是否会移动文件、启动 GPU
任务或写入外部系统。

`download_training_corpus.py` 只从公开的 Hugging Face Dataset 下载
`NiceAsiv/nuosu-corpus@v2026.08.02`，并用该版本 `manifest.json` 校验 CPT/SFT 文件哈希。
