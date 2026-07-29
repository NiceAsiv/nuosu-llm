# Command catalog / 命令目录

## 推荐入口

| 阶段 | 命令 |
|---|---|
| 模型恢复 | `bash scripts/model/recover_qwen3_base.sh` |
| 64 条过拟合门禁 | `bash scripts/gates/run_sft_overfit_64.sh /verified/model` |
| 研究训练流水线 | `bash scripts/pipeline/run_research_pipeline.sh /verified/model` |
| 数据画像 | `python scripts/profile_dataset.py --config ...` |
| 训练 | `python scripts/train.py --config ... --base-model ...` |
| 三卡评测 | `python -m torch.distributed.run ... scripts/evaluate_prompts.py ...` |
| 自动评分 | `python scripts/score_evaluation.py ...` |
| 合并 adapter | `python scripts/merge_adapter.py ...` |

## 目录职责

- `model/`：下载、实际 SHA-256 校验和未训练底模冒烟；
- `gates/`：小样本过拟合和机器可判定的停止条件；
- `pipeline/`：门禁通过后的分阶段三卡训练；
- `legacy/`：为复现实验保留的 watcher/overnight 脚本，不是新实验入口；
- 根目录 Python 脚本：当前训练、评测、语料画像与泄漏审计工具；
- `benchmark_throughput.sh`：GPU 空闲时运行的独立吞吐基准。

每个命令均支持 `--help`。自动化脚本必须在 README 中说明是否会移动文件、启动 GPU
任务或写入外部系统。
