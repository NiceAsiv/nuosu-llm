# 正式评测 / Formal Evaluation

正式结果使用 `nuosu-corpus` 中固定的 80/10/10 研究切分。保留测试集为：

```text
../nuosu-corpus/data/processed/bootstrap_nuosu_bench/research_test_eval.jsonl
```

该文件包含 11,171 条 user prompt 和 reference。reference 只用于离线评分，不会放入模型
输入。训练、选 checkpoint 和调参只能使用 train/validation；查看 test 结果后改变训练方案时，
下一次 test 分数应视为开发结果，而不是无偏的一次性结果。

The formal result uses the fixed 80/10/10 research split from `nuosu-corpus`.
The held-out file contains 11,171 user prompts and references. References are
used only by the offline scorer and are never included in model inputs. Use
train/validation for training and checkpoint selection.

## 三卡 LoRA 评测 / Three-GPU LoRA evaluation

下面的命令采用确定性 greedy decoding。每个进程只占用一张 GPU，最后由 rank 0 按原始
数据顺序合并结果。评测器同时将 Qwen 的 `<|endoftext|>` 和 ChatML 的 `<|im_end|>` 作为
停止符。

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
python -m torch.distributed.run --standalone --nproc_per_node=3 \
  scripts/evaluate_prompts.py \
  --model Qwen/Qwen3-8B-Base \
  --adapter outputs/qwen3-8b-nuosu-nuosubench-sft-3gpu-fast \
  --tokenizer outputs/qwen3-8b-nuosu-nuosubench-sft-3gpu-fast \
  --input ../nuosu-corpus/data/processed/bootstrap_nuosu_bench/research_test_eval.jsonl \
  --output evaluation/results/qwen3-8b-nuosu-research-test.jsonl \
  --batch-size 16 \
  --max-input-tokens 512 \
  --max-new-tokens 96
```

如任务中断，可保留各 rank 的 `.rankN.jsonl` 文件并在相同参数后增加 `--resume`。
先用 `--limit 192` 进行冒烟测试，确认输出语言、EOS 和加载的 adapter 正确，再消耗完整
测试集。

The evaluator performs deterministic batched generation, shards examples
across one process per GPU, and merges rank outputs in source order. Add
`--resume` after an interruption, or use `--limit 192` for a smoke test.

## 自动评分 / Automatic scoring

```bash
python scripts/score_evaluation.py \
  --input evaluation/results/qwen3-8b-nuosu-research-test.jsonl \
  --label qwen3-8b-nuosu-final \
  --output-json evaluation/results/qwen3-8b-nuosu-research-test.metrics.json \
  --output-markdown evaluation/results/qwen3-8b-nuosu-research-test.metrics.md
```

报告给出 overall 以及按目标语言 `yi`、`zh`、`unknown` 分组的：

- exact match 与忽略空白、标点后的 compact exact match；
- reference containment；
- 字符级 chrF2（越高越好）与 CER（越低越好）；
- 空输出率、彝文严格匹配率、平均生成 token 和单样本生成时间。

开放式翻译不能只看严格匹配。正式报告还应包含人工盲评：准确性、彝文规范性、流畅度各
1–5 分，并至少抽取汉译彝、彝译汉和混排任务各 100 条。自动指标为可复现基线，不替代母语
者评审。

For open-ended translation, automatic metrics are a reproducible baseline,
not a substitute for native-speaker review. A formal report should also
include blind 1–5 ratings for accuracy, orthographic correctness, and fluency,
with at least 100 samples from each major direction.

## 上游完整基准 / Full upstream benchmark

未使用 NuosuBench 训练的基础模型可下载完整上游测试集：

```bash
python scripts/fetch_nuosu_bench.py \
  --revision 0709c1fd4f6eaabd4b058ea419027ccc42731dee
```

若模型训练使用了 NuosuBench 的任何部分，不能把完整上游集分数写成无污染的外部
benchmark。完整 `test.jsonl` 主要用于未接触该数据的基线模型。

For models trained on any NuosuBench subset, do not present the full upstream
test score as an uncontaminated external benchmark. Use the reserved research
split and disclose the training source.

训练前可检查精确文本泄漏：

```bash
python scripts/check_benchmark_leakage.py \
  --benchmark evaluation/nuosu_bench/test.jsonl \
  --train ../nuosu-corpus/data/processed/cpt/train.jsonl \
          ../nuosu-corpus/data/processed/sft/train.jsonl
```

精确重叠检查不能发现改写、截断或近似重复，因此还需要分组切分、近似去重与人工抽样。
