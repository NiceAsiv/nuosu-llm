# Nuosu LLM

[![CI](https://github.com/NiceAsiv/nuosu-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/NiceAsiv/nuosu-llm/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-NiceAsiv-yellow)](https://huggingface.co/NiceAsiv)

面向凉山规范彝文（诺苏语）的可复现大语言模型适配、评测与发布工具。

Nuosu LLM is a reproducible toolkit for adapting and evaluating large language
models for Standard Liangshan Yi (Nuosu). The current research line studies
whether low-rank Nuosu adaptation can improve language capability while
preserving the reasoning behavior of a post-trained foundation model.

## 研究概览

当前开发实验以固定版本的 `Qwen/Qwen3-8B` 为起点，采用两阶段 QLoRA：

1. 使用规范彝文连续文本进行低学习率持续预训练（CPT）；
2. 使用翻译、词典、问答等统一指令数据进行 completion-only SFT。

SFT 仅在每条训练提示的最后一个用户消息中加入 `/no_think`，使短答案成为显式的上下文行为，
而不是在模型层面全局抑制思考。训练保留 Qwen3 官方聊天模板，并在 CPT、SFT 前后执行推理
回归门禁。

| 项目 | 固定设置 |
|---|---|
| 基础模型 | `Qwen/Qwen3-8B` |
| 模型 revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| 训练语料 | `NiceAsiv/nuosu-corpus@v2026.08.02` |
| CPT | 4,238 条，约 366 万 token；训练前无损切分至 2,048 token |
| SFT | 113,269 条，约 664 万 token |
| 参数高效微调 | QLoRA NF4、BF16 compute、LoRA rank 16 |
| 当前阶段 | seed 42 开发实验；正式结论需多随机种子复现和母语者盲评 |

训练 loss、公开基准分数或少量示例都不能单独证明彝语质量。本项目在完整评测和母语者审核
完成前不宣称模型已经达到可部署水平。

## 项目边界

本仓库负责模型下载与校验、训练、评测、adapter 检查和发布准备。语料采集、OCR 后处理、
清洗、来源记录与 Hugging Face 数据集构建位于独立仓库
[`NiceAsiv/nuosu-corpus`](https://github.com/NiceAsiv/nuosu-corpus)。建议将两个仓库放在同一父目录：

```text
workspace/
├── nuosu-corpus/    # 语料采集、清洗、来源元数据与发布
└── nuosu-llm/       # 模型训练、评测与发布
```

语言范围以1980年凉山规范彝文为核心，主要面向诺苏语北部方言、圣乍话基础方言和喜德标准音。
本项目不宣称覆盖全部彝语支系、传统彝文或所有地方口语。

## 安装

```bash
git clone https://github.com/NiceAsiv/nuosu-corpus.git
git clone https://github.com/NiceAsiv/nuosu-llm.git
cd nuosu-llm

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[train,dev]"
python -m pytest -q
```

只进行代码检查时可安装轻量开发依赖：

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest -q
```

## 可复现训练

以下命令展示当前8B开发实验的核心路径。完整的假设、门禁、随机种子与报告规则见
[`experiments/qwen3-8b-preserve-thinking-20260803/`](experiments/qwen3-8b-preserve-thinking-20260803/)。

### 1. 下载固定语料版本

```bash
python scripts/download_training_corpus.py
```

脚本下载 `NiceAsiv/nuosu-corpus@v2026.08.02`，并根据数据集 `manifest.json` 校验
`ready_cpt.jsonl` 和 `ready_sft.jsonl` 的记录数与 SHA-256。训练配置不会读取可变的 `main`
分支。

### 2. 下载并验证基础模型

```bash
MODEL_DIR=/absolute/path/to/models/Qwen3-8B-b968826d

python scripts/model/download_snapshot.py \
  --repo-id Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --output-dir "${MODEL_DIR}"

(cd "${MODEL_DIR}" && sha256sum --check VERIFIED.sha256)
python scripts/model/smoke_test_base.py --model "${MODEL_DIR}"
python scripts/model/check_qwen3_template.py \
  --tokenizer "${MODEL_DIR}" \
  --output artifacts/template-gate.json
```

训练入口只接受带有 `VERIFIED.sha256` 的本地模型目录。固定 revision、实际文件哈希和未训练
模型生成冒烟共同防止不完整权重进入训练链。

### 3. 构建无损 CPT 训练视图

直接截断长记录会丢失语料。先按实际 tokenizer 将连续文本无损切分：

```bash
python scripts/chunk_cpt.py \
  --input ../nuosu-corpus/data/hf/nuosu-corpus/ready_cpt.jsonl \
  --output artifacts/cpt/ready_cpt.chunks.jsonl \
  --tokenizer "${MODEL_DIR}" \
  --max-tokens 2048 \
  --source-revision v2026.08.02 \
  --tokenizer-revision b968826d9c46dd6066d109eabc6255188de91218
```

生成的 manifest 会记录源文件哈希、切分数量、token 上限和字符重建检查。

### 4. CPT

单卡运行：

```bash
python scripts/train.py \
  --config configs/cpt_qwen3_8b_qlora.yaml \
  --base-model "${MODEL_DIR}" \
  --train-file artifacts/cpt/ready_cpt.chunks.jsonl
```

双卡运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train.py \
  --config configs/cpt_qwen3_8b_qlora.yaml \
  --base-model "${MODEL_DIR}" \
  --train-file artifacts/cpt/ready_cpt.chunks.jsonl
```

CPT adapter 只有在固定推理 canary 的答案保持率、完整思考率和人工核查均通过后，才能传递
给 SFT。八条 canary 只用于运行安全检查，不构成论文级推理能力结论。

### 5. SFT

`configs/sft_qwen3_8b_qlora.yaml` 默认从 CPT adapter 继续训练，并对113,269条记录执行一轮
全量 completion-only SFT：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/train.py \
  --config configs/sft_qwen3_8b_qlora.yaml \
  --base-model "${MODEL_DIR}"
```

不同 GPU 数量可以使用同一入口，但应重新核对有效全局 batch：

```text
world_size × per_device_train_batch_size × gradient_accumulation_steps
```

较小算力环境可参考 [`recipes/qwen3-1.7b-full/`](recipes/qwen3-1.7b-full/) 的1.7B两阶段配方。

## 评测原则

至少比较以下模型状态：

- `M0`：冻结的官方后训练 Qwen3-8B；
- `M2`：M0 + Nuosu CPT adapter；
- `M3`：通过 CPT 门禁后完成上下文化 `/no_think` SFT 的模型。

评测分为四类：

- 彝汉翻译、规范书写、问答与指令遵循；
- 空输出、乱码、异常重复、EOS 和长度截断；
- GSM8K、MMLU-Pro、IFEval 等通用能力回归；
- 母语者对正确性、流利度、规范性和幻觉的盲评。

当前低资源策略将所有可用发布语料用于训练，因此 NuosuBench 与训练来源可能存在文本级或
作品级重合。其结果只能作为披露重合后的诊断指标，不能描述为无污染测试。论文级比较应在
冻结训练配方后使用独立盲评集，并报告多随机种子结果、均值、标准差和置信区间。

详见 [`evaluation/README.md`](evaluation/README.md) 和
[`docs/04-evaluation.md`](docs/04-evaluation.md)。

## 使用 adapter

```bash
CUDA_VISIBLE_DEVICES=0 nuosu-chat \
  --model /absolute/path/to/verified-model \
  --adapter /absolute/path/to/adapter
```

输入 `/clear` 清空多轮上下文，输入 `/quit` 退出；显存受限时可增加 `--load-in-4bit`。
不懂彝文的测试者可以检查运行时行为和格式，但不能代替母语者判断语义与书写规范性。

## 文档与命令索引

| 需求 | 文档或入口 |
|---|---|
| 范围与模型路线 | [`docs/01-scope-and-architecture.md`](docs/01-scope-and-architecture.md) |
| 配置说明 | [`configs/README.md`](configs/README.md) |
| 训练、恢复与多卡 | [`docs/03-training-guide.md`](docs/03-training-guide.md) |
| 评测协议 | [`evaluation/README.md`](evaluation/README.md) |
| 部署与 adapter 合并 | [`docs/05-deployment.md`](docs/05-deployment.md) |
| 工具索引 | [`scripts/README.md`](scripts/README.md) |
| 实验记录 | [`experiments/README.md`](experiments/README.md) |
| 失败复盘与保护措施 | [`docs/07-lessons-learned.md`](docs/07-lessons-learned.md) |
| 发布规范 | [`docs/09-publishing.md`](docs/09-publishing.md) |

运行产物统一写入 Git 忽略的 `outputs/`、`artifacts/`、`logs/` 和
`evaluation/results/`。一次性事故恢复工具保留在 [`patches/`](patches/)，不属于正常训练
入口。

## 开发与贡献

```bash
make test
make lint
```

新增训练配置或实验报告时，请同时记录：

- 模型 ID、revision 与逐文件校验结果；
- 数据仓库 revision、文件哈希和样本统计；
- 代码 commit、随机种子、依赖版本和硬件；
- 完整训练参数、门禁结果和失败记录；
- 数据重合、评测限制与人工审核协议。

## 引用

如果本仓库对你的研究有帮助，请引用：

```bibtex
@software{axi2026nuosullm,
  author      = {Wuhe Axi},
  title       = {Nuosu LLM: Reproducible Adaptation and Evaluation for Standard Liangshan Yi},
  year        = {2026},
  institution = {Xi'an Jiaotong University},
  url         = {https://github.com/NiceAsiv/nuosu-llm}
}
```

语料使用还应单独引用 [`NiceAsiv/nuosu-corpus`](https://github.com/NiceAsiv/nuosu-corpus)
及其列出的原始资料。

## 许可证

代码以 [Apache License 2.0](LICENSE) 发布。模型、语料和第三方资料可能适用各自的许可证
与使用条件；发布或再分发前请分别核对。
