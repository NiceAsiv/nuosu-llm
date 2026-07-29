# Nuosu LLM / 诺苏语大模型

[中文](#中文) · [English](#english)

## 中文

### 项目定位

本仓库基于 [`Qwen/Qwen3-8B-Base`](https://huggingface.co/Qwen/Qwen3-8B-Base) 开发面向
凉山规范彝文 / 诺苏语的模型训练、评测、adapter 合并和部署流程。语料采集、来源记录、方言标签、
清洗和格式转换位于相邻的 `nuosu-corpus` 仓库。

第一阶段只面向诺苏语北部方言、圣乍话基础方言、喜德标准音和 1980 年凉山规范彝文。项目不宣称
支持所有彝语支系或传统彝文字形。

### 仓库边界

```text
nuosu-corpus/                 # 数据仓库
├── data/
├── scripts/
├── src/nuosu_corpus/
└── docs/

nuosu-llm/                    # 模型仓库
├── configs/                  # Qwen3-8B CPT / SFT 配置
├── docker/
├── evaluation/
├── scripts/                  # 训练、评测、泄漏检查、合并
├── src/nuosu_llm/
└── docs/
```

两个仓库默认位于同一父目录，训练配置通过
`../nuosu-corpus/data/processed/...` 引用可复现的数据产物。

### 模型与数据来源

基础模型：

- [Qwen/Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base)，8.2B 参数，
  Apache-2.0，32,768 token context。

当前数据来源：

- [彝汉电子词典](https://www.yixueyanjiu.com/dict/)：词典翻译 SFT；
- [TianYeZ1214/NuosuBench](https://huggingface.co/datasets/TianYeZ1214/NuosuBench)：
  80/10/10 学术研究 train/validation/test；
- [isljsy/yidir](https://github.com/isljsy/yidir)：喜德读音研究，PUA 未映射，不进入训练；
- [NuosuBburma OCR Evaluation Set](https://huggingface.co/datasets/nanxidajun/NuosuBburma-OCR-Evaluation-Set)：
  整页人工 GT 用于连续文本 CPT，并按原始文献切分。

完整来源和语料现状见 sibling 仓库的 `README.md` 与
`docs/11-corpus-readiness-audit.md`。

### Benchmark 策略

NuosuBench 官方只有一个 `test` split，共 110,513 条。本项目按内容 hash 稳定切成：

- train：88,487；
- validation：10,839；
- internal research test：11,171。

训练配置只使用 train 和 validation，最终研究测试只使用保留的 internal test。完整官方
`test.jsonl` 仍可用于测试未适配的基础模型，但适配后的模型以保留 research test 为准。

OCR 整页 GT 也按 `source_title` 做文档级切分：train 398、validation 57、test 64。区域裁剪
不重复加入训练。

### 当前语料是否足够

目前已经能启动学术研究训练，但要形成可靠的通用模型仍需补齐：

- 词典与 NuosuBench 都以单轮翻译/问答为主；
- 多轮对话记录为 0；
- 目前只有 398 条 OCR 整页文本用于 CPT，规模仍小；
- 没有项目自建的母语者盲评集。

建议先完成 OCR GT CPT + 词典/NuosuBench SFT 的研究基线，再优先补充文章和多轮对话。

### 安装

```bash
git clone https://github.com/NiceAsiv/nuosu-corpus.git
git clone https://github.com/NiceAsiv/nuosu-llm.git

cd nuosu-llm
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[train,dev]"
pytest -q
```

### 训练

继续预训练：

```bash
python scripts/train.py --config configs/cpt_qwen3_8b_qlora.yaml
```

普通 SFT：

```bash
python scripts/train.py --config configs/sft_qwen3_8b_qlora.yaml
```

词典研究训练：

```bash
python scripts/train.py \
  --config configs/sft_qwen3_8b_dictionary_research.yaml
```

NuosuBench 学术研究训练：

```bash
python scripts/train.py \
  --config configs/sft_qwen3_8b_bootstrap.yaml
```

OCR GT 连续文本 CPT：

```bash
python scripts/train.py \
  --config configs/cpt_qwen3_8b_ocr_gt_research.yaml
```

研究结果应注明使用的数据源、revision 和保留测试集。

服务器路径不同可覆盖数据位置：

```bash
python scripts/train.py \
  --config configs/cpt_qwen3_8b_qlora.yaml \
  --train-file /data0/nuosu-corpus/cpt/train.jsonl \
  --eval-file /data0/nuosu-corpus/cpt/validation.jsonl \
  --output-dir /data0/nuosu-checkpoints/qwen3-8b-cpt
```

### 多卡吞吐优化

三卡训练使用数据并行：每张卡处理不同的 micro-batch，反向完成后聚合梯度。短样本不能只把
`nproc_per_node` 改为 3；如果每卡仍然只有一条几十 token 的样本，GPU 会长期等待 Python、
kernel launch 和 PCIe 梯度同步。

先画像，再基准，最后正式训练：

```bash
# 统计 p50/p90/p99/max token 长度、截断数和分桶效率
python scripts/profile_dataset.py \
  --config configs/sft_qwen3_8b_dictionary_research_3gpu_fast.yaml \
  --batch-size 32

# GPU 空闲时执行独立的 30-step 三卡吞吐基准
bash scripts/benchmark_throughput.sh \
  configs/sft_qwen3_8b_dictionary_research_3gpu_fast.yaml \
  30 \
  outputs/previous-adapter-or-checkpoint
```

`benchmark_throughput.sh` 检测到已有 GPU 计算进程时会拒绝启动，基准结果写入
`logs/benchmarks/`，模型写入 `outputs/benchmarks/`。仓库提供的 `*_3gpu_fast.yaml` 使用
BF16、SDPA、长度分桶、更大的单卡 batch、fused AdamW、后台数据加载和较低的
eval/checkpoint 频率。batch 应由真实 token 长度和短基准决定，不能仅按样本条数猜测。

串行运行 OCR CPT、词典 SFT 和 NuosuBench SFT：

```bash
DICTIONARY_INIT_ADAPTER=outputs/existing-dictionary-checkpoint \
  bash scripts/run_overnight_pipeline.sh
```

若不设置 `DICTIONARY_INIT_ADAPTER`，词典阶段从配置文件中的 `init_adapter` 开始。词典成功后，
流水线才会启动 NuosuBench；任何阶段失败都会终止流水线。

NuosuBench 会先按 512 token 分成短样本主阶段和长尾补充阶段。这样短样本保持较大的单卡
batch，长样本不会因为主阶段的 512-token 上限而消失：

```bash
python scripts/bucket_sft_by_length.py \
  --config configs/sft_qwen3_8b_nuosubench_research_3gpu.yaml \
  --threshold 512 \
  --output-dir ../nuosu-corpus/data/processed/bootstrap_nuosu_bench_length_buckets
```

分桶清单会记录每个 split 的总数、短/长样本数和最长 token 数。长尾阶段在独立配置中使用小
batch，并继续训练短样本阶段的 adapter。

### 使用 Hugging Face 语料或自建语料

下游用户不必保持两个仓库相邻。可以先从 Hugging Face 下载发布版语料，并固定实际使用的
revision：

```bash
hf download YOUR_ORG/nuosu-corpus \
  --repo-type dataset \
  --revision DATASET_COMMIT_OR_TAG \
  --local-dir /data0/nuosu-corpus
```

下载后只需覆盖输入文件和输出目录。CPT 文件每行应为 `{"text": "..."}`，SFT 文件每行应为
`{"messages": [...]}`；两种格式均可带 `id` 和 `metadata`。

使用发布版或自建 CPT 语料：

```bash
python scripts/train.py \
  --config configs/cpt_qwen3_8b_qlora.yaml \
  --train-file /data0/nuosu-corpus/my_cpt/train.jsonl \
  --eval-file /data0/nuosu-corpus/my_cpt/validation.jsonl \
  --output-dir /data0/checkpoints/qwen3-8b-my-cpt
```

使用发布版或自建 SFT 语料：

```bash
python scripts/train.py \
  --config configs/sft_qwen3_8b_qlora.yaml \
  --train-file /data0/nuosu-corpus/my_sft/train.jsonl \
  --eval-file /data0/nuosu-corpus/my_sft/validation.jsonl \
  --output-dir /data0/checkpoints/qwen3-8b-my-sft
```

自建语料的清洗、分组切分和校验由 `nuosu-corpus` 仓库的 `prepare_corpus.py`、
`prepare_sft.py` 与 `validate_dataset.py` 完成。推荐完整流程为：

1. 构建并校验 train/validation/test；
2. 记录 Dataset ID、revision、构建参数和数据统计；
3. 用 train/validation 做 CPT；
4. 合并 CPT adapter，得到新的本地基础模型；
5. 将 SFT 配置中的 `base_model` 改为合并后的 CPT 模型路径，再做 SFT；
6. 只在模型和配置确定后运行保留 test。

若目标是 “CPT 后继续 SFT”，可先合并 CPT adapter 并把合并目录设为 SFT 的
`base_model`，也可在 SFT 配置中设置 `init_adapter`，继续训练同一个 LoRA adapter。不要让
SFT 在没有 `init_adapter` 的情况下重新从 `Qwen/Qwen3-8B-Base` 开始。

### 评测

```bash
python scripts/fetch_nuosu_bench.py \
  --revision 0709c1fd4f6eaabd4b058ea419027ccc42731dee

python scripts/evaluate_prompts.py \
  --model Qwen/Qwen3-8B-Base \
  --adapter outputs/qwen3-8b-nuosu-nuosubench-sft-3gpu-fast \
  --tokenizer outputs/qwen3-8b-nuosu-nuosubench-sft-3gpu-fast \
  --input ../nuosu-corpus/data/processed/bootstrap_nuosu_bench/research_test_eval.jsonl \
  --output evaluation/results/qwen3-8b-nuosu-research-test.jsonl

python scripts/score_evaluation.py \
  --input evaluation/results/qwen3-8b-nuosu-research-test.jsonl \
  --label qwen3-8b-nuosu-final \
  --output-json evaluation/results/qwen3-8b-nuosu-research-test.metrics.json \
  --output-markdown evaluation/results/qwen3-8b-nuosu-research-test.metrics.md
```

多 GPU、断点续跑、指标解释和人工盲评要求见
[`evaluation/README.md`](evaluation/README.md)。

当前实验 checkpoint 的完整结果与下一轮门禁方案见
[`evaluation/reports/2026-07-29-qwen3-8b-nuosu.md`](evaluation/reports/2026-07-29-qwen3-8b-nuosu.md)。

完整 `evaluation/nuosu_bench/test.jsonl` 主要用于比较未使用 NuosuBench 训练的基础模型。研究适配
模型以 `research_test_eval.jsonl` 为最终保留测试集。训练前可运行精确重叠检查：

```bash
python scripts/check_benchmark_leakage.py \
  --benchmark evaluation/nuosu_bench/test.jsonl \
  --train ../nuosu-corpus/data/processed/cpt/train.jsonl \
          ../nuosu-corpus/data/processed/sft/train.jsonl
```

精确检查只是底线，正式训练还需要来源级隔离、MinHash/语义近重复和人工抽样。

### 合并与学术引用

```bash
python scripts/merge_adapter.py \
  --base-model Qwen/Qwen3-8B-Base \
  --adapter outputs/qwen3-8b-sft-qlora \
  --output outputs/qwen3-8b-nuosu-merged
```

发布 adapter 或合并模型：

```bash
hf auth login
hf upload YOUR_ORG/nuosu-model outputs/qwen3-8b-nuosu-merged . \
  --repo-type model
```

模型卡至少应记录基础模型、语料 Dataset ID 与 revision、CPT/SFT 顺序、完整训练配置、
随机种子和保留测试结果。Hugging Face 的官方上传说明见
[Upload files to the Hub](https://huggingface.co/docs/huggingface_hub/guides/upload)。

- 代码仓库发布到 GitHub；
- 数据和模型产物可发布到 Hugging Face；
- 论文、报告和模型卡应引用 Qwen3、NuosuBench、yidir、彝汉电子词典和 OCR GT 数据集；
- 明确给出数据 revision、切分方式和训练配置。

### 文档

- [`docs/01-scope-and-architecture.md`](docs/01-scope-and-architecture.md)
- [`docs/03-training-guide.md`](docs/03-training-guide.md)
- [`docs/04-evaluation.md`](docs/04-evaluation.md)
- [`docs/05-deployment.md`](docs/05-deployment.md)
- [`docs/06-training-server.md`](docs/06-training-server.md)
- [`docs/09-publishing.md`](docs/09-publishing.md)

## English

### Overview

This repository provides training, evaluation, adapter merging, and deployment workflows for a Standard
Liangshan Yi / Nuosu model based on
[`Qwen/Qwen3-8B-Base`](https://huggingface.co/Qwen/Qwen3-8B-Base). Corpus collection, provenance,
source tracking, dialect labeling, cleaning, and conversion live in the sibling `nuosu-corpus`
repository.

The first release targets Nuosu in the Northern Yi branch, the Shynra reference dialect, Xide standard
pronunciation, and the Standard Liangshan Yi orthography approved in 1980. It does not claim coverage of
all Yi varieties or traditional scripts.

### Base model and data sources

- Base model: [Qwen/Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base), 8.2B parameters,
  Apache-2.0, 32,768-token context.
- [Yi–Chinese electronic dictionary](https://www.yixueyanjiu.com/dict/): dictionary translation SFT.
- [NuosuBench](https://huggingface.co/datasets/TianYeZ1214/NuosuBench): 80/10/10 academic
  train/validation/test split.
- [yidir](https://github.com/isljsy/yidir): Xide pronunciation research; PUA glyphs are not mapped.
- [NuosuBburma OCR Evaluation Set](https://huggingface.co/datasets/nanxidajun/NuosuBburma-OCR-Evaluation-Set):
  page-level human GT for document-split CPT.

### Benchmark isolation

NuosuBench contains 110,513 records in one upstream split. The academic workflow deterministically
reserves 88,487 for training, 10,839 for validation, and 11,171 for an internal research test. Training
uses only the first two partitions and final adapted-model evaluation uses the reserved research test.

OCR page GT is split by complete source title into 398 train, 57 validation, and 64 test pages. Region
crops are excluded to avoid duplicate supervision.

### Readiness

The current data is sufficient to start an academic baseline, but not yet a reliable general-purpose
model. Existing SFT is dominated by single-turn translation, there are no multi-turn conversations, the
long-form CPT set has only 398 training pages, and there is no project-owned native-speaker blind test.

### Quick start

```bash
git clone https://github.com/NiceAsiv/nuosu-corpus.git
git clone https://github.com/NiceAsiv/nuosu-llm.git
cd nuosu-llm

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[train,dev]"
pytest -q
```

### Training with a published or custom corpus

The data repository can live anywhere. Download a pinned Hugging Face Dataset revision:

```bash
hf download YOUR_ORG/nuosu-corpus \
  --repo-type dataset \
  --revision DATASET_COMMIT_OR_TAG \
  --local-dir /data0/nuosu-corpus
```

CPT JSONL uses one `{"text": "..."}` record per line. SFT JSONL uses one
`{"messages": [...]}` record per line. Both may include `id` and `metadata`.

Train CPT with downloaded or locally built data:

```bash
python scripts/train.py \
  --config configs/cpt_qwen3_8b_qlora.yaml \
  --train-file /data0/nuosu-corpus/my_cpt/train.jsonl \
  --eval-file /data0/nuosu-corpus/my_cpt/validation.jsonl \
  --output-dir /data0/checkpoints/qwen3-8b-my-cpt
```

Train SFT in the same way:

```bash
python scripts/train.py \
  --config configs/sft_qwen3_8b_qlora.yaml \
  --train-file /data0/nuosu-corpus/my_sft/train.jsonl \
  --eval-file /data0/nuosu-corpus/my_sft/validation.jsonl \
  --output-dir /data0/checkpoints/qwen3-8b-my-sft
```

### Multi-GPU throughput

Three-GPU runs use data parallelism: every GPU processes a different micro-batch and gradients are
aggregated after backward. Very short records remain inefficient if each GPU receives only one record.
Profile token lengths first, then run an isolated short benchmark:

```bash
python scripts/profile_dataset.py \
  --config configs/sft_qwen3_8b_dictionary_research_3gpu_fast.yaml \
  --batch-size 32

bash scripts/benchmark_throughput.sh \
  configs/sft_qwen3_8b_dictionary_research_3gpu_fast.yaml \
  30 \
  outputs/previous-adapter-or-checkpoint
```

The benchmark refuses to start while another GPU compute process is active. The `*_3gpu_fast.yaml`
profiles use BF16, SDPA, length grouping, larger per-device batches, fused AdamW, background data
loading, and less frequent evaluation/checkpointing. Always select batch size from measured token
lengths and a short stability benchmark.

For datasets with a small long-context tail, `bucket_sft_by_length.py` creates reproducible short and
long JSONL buckets plus a manifest. The main stage keeps a large batch for short rows, while a
follow-up low-batch stage continues the same adapter on long rows instead of silently truncating them.

Build and validate custom splits with `prepare_corpus.py`, `prepare_sft.py`, and
`validate_dataset.py` in the `nuosu-corpus` repository. Record the Dataset ID, exact revision, build
parameters, and statistics before training.

For sequential CPT then SFT, either merge the CPT adapter and use the merged model as `base_model`, or
set `init_adapter` in the SFT configuration to continue training the same LoRA adapter. Omitting both
starts an independent adapter from `Qwen/Qwen3-8B-Base`.

Fetch the evaluation benchmark:

```bash
python scripts/fetch_nuosu_bench.py \
  --revision 0709c1fd4f6eaabd4b058ea419027ccc42731dee
```

Run the reserved research test:

```bash
python scripts/evaluate_prompts.py \
  --model outputs/qwen3-8b-nuosu-merged \
  --input ../nuosu-corpus/data/processed/bootstrap_nuosu_bench/research_test_eval.jsonl \
  --output evaluation/results/qwen3-8b-nuosu-research-test.jsonl
```

### Academic attribution

Publish code on GitHub and data/model artifacts on Hugging Face as appropriate. Papers, reports, and
model cards should cite Qwen3 and all four corpus sources, and state the exact revisions, split policy,
and training configuration.

Publish an adapter or merged model with:

```bash
hf auth login
hf upload YOUR_ORG/nuosu-model outputs/qwen3-8b-nuosu-merged . \
  --repo-type model
```

The Model Card should include the base model, corpus Dataset ID and revision, CPT/SFT order, complete
configuration, seed, and held-out test results. See the official
[Hugging Face upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload).
