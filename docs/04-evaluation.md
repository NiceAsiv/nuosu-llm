# 评测方案

## 1. 评测原则

诺苏语是低资源语言，自动指标覆盖有限。正式结论必须由母语者评测支持。

评测至少比较：

1. 原始基础模型；
2. CPT 后模型；
3. CPT + SFT 后模型；
4. 最终 INT4 模型。

量化模型必须单独评测，不能假设它与 BF16 合并模型完全一致。

## 2. 评测集结构

建议首轮 300～500 条：

| 类别 | 建议数量 |
|---|---:|
| 诺苏语理解与问答 | 80 |
| 汉译诺苏 | 60 |
| 诺苏译汉 | 60 |
| 摘要、改写和纠错 | 50 |
| 文化、地名和专有名词 | 40 |
| 拒答、安全和不确定性 | 30 |
| 多轮上下文 | 30 |

扩展评测集需要覆盖：

- 正式书面语与口语表达；
- 短句与长文；
- 数字、日期、人名和地名；
- 汉语夹杂、拉丁转写和错别字；
- 不同来源和主题；
- 易混方言或书写体系。

## 3. 人工评分

每项使用 1～5 分：

| 指标 | 1 分 | 3 分 | 5 分 |
|---|---|---|---|
| 准确性 | 含义基本错误 | 核心含义正确但有遗漏 | 准确完整 |
| 自然度 | 不像母语表达 | 可以理解但生硬 | 自然流畅 |
| 书写规范 | 大量错字/乱码 | 少量规范问题 | 符合规范 |
| 任务遵循 | 未完成任务 | 部分完成 | 完整遵循 |
| 文化恰当性 | 明显不当 | 基本可接受 | 恰当且尊重语境 |

另外记录：

- 是否出现幻觉；
- 是否无故切换为汉语；
- 是否包含其他彝语支系但未说明；
- 是否泄露训练数据；
- 是否需要人工修改后才能使用。

## 4. 评审流程

- 至少一名诺苏语母语者逐条评分；
- 高风险、低一致性样本由第二名评审复核；
- 模型名称对评审者隐藏；
- 随机打乱输出顺序；
- 保存分歧，不强行平均掉语言差异；
- 计算评分均值、分布和评审一致性。

## 5. 自动指标

可作为辅助信号：

- CPT eval loss / perplexity；
- 翻译 chrF、BLEU；
- 字符错误率；
- 输出中的规范彝文字符比例；
- `<unk>`、乱码和异常重复率；
- 中文通用保持集准确率；
- 延迟、吞吐、显存和内存占用。

BLEU 等指标对低资源语言、自由表达和多种合理翻译尤其不稳定，不能单独用于上线决策。

## 6. 固定提示集

输入 JSONL：

```json
{
  "id": "eval-0001",
  "category": "qa",
  "messages": [
    {"role": "user", "content": "经过核验的测试提示"}
  ]
}
```

生成待评分文件：

```bash
python scripts/evaluate_prompts.py \
  --model outputs/qwen3-8b-nuosu-merged \
  --input evaluation/prompts.jsonl \
  --output evaluation/results/model-a.jsonl
```

脚本只生成模型回答和基础语言比例。人工评分字段必须由评审流程填写。

## 7. NuosuBench

`TianYeZ1214/NuosuBench` 包含 110,513 条标准彝文样本，官方只提供 `test` split。它覆盖词汇、句子、史诗、公文和教育五类能力。

```bash
python scripts/fetch_nuosu_bench.py

python scripts/check_benchmark_leakage.py \
  --benchmark evaluation/nuosu_bench/test.jsonl \
  --train ../nuosu-corpus/data/processed/cpt/train.jsonl \
          ../nuosu-corpus/data/processed/sft/train.jsonl
```

学术研究训练在 `nuosu-corpus` 中按内容 hash 固定切成 80% train、10% validation 和 10%
internal research test。训练后使用
`../nuosu-corpus/data/processed/bootstrap_nuosu_bench/research_test_eval.jsonl` 评测，不能直接
把包含 assistant 答案的 SFT `internal_test.jsonl` 作为生成输入。完整 `test.jsonl` 主要用于
未使用 NuosuBench 训练的基础模型比较。

## 8. 上线门槛示例

首版门槛应由项目团队和社区代表共同确认。可采用：

- 规范书写平均分 ≥ 4.0；
- 任务遵循平均分 ≥ 4.0；
- 明显幻觉率低于约定阈值；
- 无乱码和系统性语言代码混淆；
- INT4 相对 BF16 的人工评分下降不超过约定值；
- 服务在目标并发下满足内存和延迟要求。
