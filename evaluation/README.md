# 正式评测 / Formal Evaluation

当前训练快照 `NiceAsiv/nuosu-corpus@v2026.08.02` 采用低资源全量训练策略，不在发布语料中
预留 validation/test。历史报告所用的 80/10/10 NuosuBench 派生切分只用于复现旧实验，
不再作为当前训练或正式评测协议。

正式模型选择需要项目另行维护、未进入训练与提示构造流程的母语者盲评集。没有这样的评测集时，
可以报告格式、乱码、停止行为和公开基准上的诊断指标，但不能将其描述为无污染能力估计。

The pinned training snapshot uses all usable data and therefore does not ship a
held-out validation or test split. Formal model selection requires a separately
maintained native-speaker blind set that never enters training or prompt construction.

## 生成与自动评分

先用独立评测 JSONL 生成回答：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
torchrun --standalone --nproc_per_node=3 \
  scripts/evaluate_prompts.py \
  --model /absolute/path/to/verified-model \
  --adapter outputs/qwen3-1.7b-nuosu-sft \
  --tokenizer outputs/qwen3-1.7b-nuosu-sft \
  --input /absolute/path/to/blind-evaluation.jsonl \
  --output evaluation/results/nuosu-blind.jsonl \
  --batch-size 16 \
  --max-input-tokens 512 \
  --max-new-tokens 96
```

随后计算可复现的自动指标：

```bash
python scripts/score_evaluation.py \
  --input evaluation/results/nuosu-blind.jsonl \
  --label nuosu-blind \
  --output-json evaluation/results/nuosu-blind.metrics.json \
  --output-markdown evaluation/results/nuosu-blind.metrics.md
```

开放式翻译不能只看严格匹配。正式报告还应包含母语者对准确性、彝文规范性和流畅度的盲评。

## NuosuBench 诊断与重合审计

```bash
python scripts/fetch_nuosu_bench.py \
  --revision 0709c1fd4f6eaabd4b058ea419027ccc42731dee

python scripts/check_benchmark_leakage.py \
  --benchmark evaluation/nuosu_bench/test.jsonl \
  --train ../nuosu-corpus/data/hf/nuosu-corpus/ready_cpt.jsonl \
          ../nuosu-corpus/data/hf/nuosu-corpus/ready_sft.jsonl
```

当前语料保留作品级重合并将其记录为审计元数据，所以 NuosuBench 结果应明确披露重合风险，
不得称为无污染外部 benchmark。精确重叠检查也无法发现改写、截断和近似重复。

2026-07-29 的报告保留在 `evaluation/reports/`，仅代表当时的数据与模型快照。
