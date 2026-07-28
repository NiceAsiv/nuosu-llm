# 评测数据

`NuosuBench` 的完整上游版本可用于基础模型评测。下载后的文件默认被 Git 忽略。

```bash
python scripts/fetch_nuosu_bench.py
```

输出：

```text
evaluation/nuosu_bench/test.jsonl
evaluation/nuosu_bench/manifest.json
```

学术研究适配模型使用 `nuosu-corpus` 中稳定的 80/10/10 切分。最终测试输入为：

```text
../nuosu-corpus/data/processed/bootstrap_nuosu_bench/research_test_eval.jsonl
```

该文件只包含保留的 11,171 条 user prompt 和 reference，不会把 assistant 参考答案放进模型输入。
完整 `test.jsonl` 主要用于比较未使用 NuosuBench 训练的基础模型。

在训练前检查精确文本泄漏：

```bash
python scripts/check_benchmark_leakage.py \
  --benchmark evaluation/nuosu_bench/test.jsonl \
  --train ../nuosu-corpus/data/processed/cpt/train.jsonl \
          ../nuosu-corpus/data/processed/sft/train.jsonl
```

精确重叠检查不能发现改写、截断或近似重复，因此还需要抽样和近似去重。
