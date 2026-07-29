# Command catalog / 命令目录

## 推荐入口

| 阶段 | 命令 |
|---|---|
| 模型下载与校验 | 见 `scripts/model/README.md` |
| 64 条过拟合门禁 | `bash scripts/gates/run_sft_overfit_64.sh /verified/model` |
| 数据画像 | `python scripts/profile_dataset.py --config ...` |
| 单卡交互试用 | `nuosu-chat --model /verified/model --adapter /adapter` |
| 训练 | `python scripts/train.py --config ... --base-model ...` |
| 多卡训练 | `torchrun --nproc_per_node=N scripts/train.py ...` |
| 多卡评测 | `torchrun --nproc_per_node=N scripts/evaluate_prompts.py ...` |
| 自动评分 | `python scripts/score_evaluation.py ...` |
| 合并 adapter | `python scripts/merge_adapter.py ...` |

## 目录职责

- `model/`：下载、实际 SHA-256 校验和未训练底模冒烟；
- `gates/`：小样本过拟合和机器可判定的停止条件；
- 根目录 Python 脚本：当前训练、评测、语料画像与泄漏审计工具；
- `benchmark_throughput.sh`：GPU 空闲时运行的独立吞吐基准。

特定服务器编排位于 `experiments/`，事故恢复位于 `patches/`。它们不是稳定命令目录的一部分。

每个命令均支持 `--help`。自动化脚本必须在 README 中说明是否会移动文件、启动 GPU
任务或写入外部系统。
