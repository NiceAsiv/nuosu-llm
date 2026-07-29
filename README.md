# Nuosu LLM

[![CI](https://github.com/NiceAsiv/nuosu-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/NiceAsiv/nuosu-llm/actions/workflows/ci.yml)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/NiceAsiv)

面向凉山规范彝文（诺苏语）的可复现训练与评测工具。

> **当前状态（2026-07-29）**
>
> 第一轮实验使用了内容损坏的 `Qwen3-8B-Base` 缓存，因此该轮所有 adapter 和测试分数均
> 无效，不能发布。仓库现已把模型实际 SHA-256 校验和生成冒烟测试设为训练前置门禁。

## 从这里开始

按你的角色选择入口：

| 目标 | 入口 |
|---|---|
| 第一次运行项目 | 本页的“安全快速开始” |
| 理解项目边界与模型路线 | [`docs/01-scope-and-architecture.md`](docs/01-scope-and-architecture.md) |
| 准备或替换语料 | sibling 仓库 [`NiceAsiv/nuosu-corpus`](https://github.com/NiceAsiv/nuosu-corpus) |
| 选择训练配置 | [`configs/README.md`](configs/README.md) |
| 查找命令行工具 | [`scripts/README.md`](scripts/README.md) |
| 训练与恢复 | [`docs/03-training-guide.md`](docs/03-training-guide.md) |
| 正式评测 | [`evaluation/README.md`](evaluation/README.md) |
| 查看特定硬件实验 | [`experiments/README.md`](experiments/README.md) |
| 查看事故补丁 | [`patches/README.md`](patches/README.md) |
| 阅读失败复盘 | [`docs/07-lessons-learned.md`](docs/07-lessons-learned.md) |
| 查看文档地图 | [`docs/README.md`](docs/README.md) |

## 安全快速开始

### 1. 安装

```bash
git clone https://github.com/NiceAsiv/nuosu-corpus.git
git clone https://github.com/NiceAsiv/nuosu-llm.git
cd nuosu-llm

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[train,dev]"
pytest -q
```

两个仓库默认放在同一父目录：

```text
workspace/
├── nuosu-corpus/    # 采集、清洗、方言标签、切分、发布
└── nuosu-llm/       # 模型获取、训练、评测、合并、部署
```

### 2. 下载并验证基础模型

下载固定 revision：

```bash
MODEL_DIR=/absolute/path/to/models/Qwen3-8B-Base-49e3418fbbbc
python scripts/model/download_snapshot.py \
  --repo-id Qwen/Qwen3-8B-Base \
  --revision 49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --output-dir "${MODEL_DIR}"
```

验证实际权重并运行未训练底模冒烟：

```bash
(cd "${MODEL_DIR}" && \
  sha256sum --check /absolute/path/to/nuosu-llm/scripts/model/manifests/qwen3-8b-base-49e3418.sha256)
python scripts/model/smoke_test_base.py --model "${MODEL_DIR}"
cp scripts/model/manifests/qwen3-8b-base-49e3418.sha256 "${MODEL_DIR}/VERIFIED.sha256"
```

如果已经遇到与 2026-07-29 相同的坏缓存事故，需要隔离缓存和派生 adapter，再使用
[`patches/2026-07-29/`](patches/2026-07-29/)；正常用户不需要运行事故补丁。

### 3. 检查语料

```bash
python scripts/profile_dataset.py \
  --config configs/sft_qwen3_8b_dictionary_research.yaml \
  --batch-size 32
```

语料必须先由 `nuosu-corpus` 生成并校验。训练仓库不负责网页采集，也不应直接修改
`data/processed`。

### 4. 只在门禁通过后训练

训练时显式传入已经完成校验的本地模型目录：

```bash
python scripts/train.py \
  --config configs/cpt_qwen3_8b_ocr_gt_research.yaml \
  --base-model /absolute/path/to/models/Qwen3-8B-Base-49e3418fbbbc
```

正式大规模训练前还必须通过：

- 64 条 SFT 过拟合测试；
- `<|im_end|>` 正常停止；
- 192 条 validation 生成无乱码、无大面积长度截断。

只运行 64 条门禁：

```bash
bash scripts/gates/run_sft_overfit_64.sh "${VERIFIED_MODEL}"
```

默认入口不限定 GPU 数量。单卡直接运行上面的命令；多卡使用实际可用数量启动同一个
`scripts/train.py`：

```bash
torchrun --standalone --nproc_per_node=NUM_GPUS \
  scripts/train.py \
  --config configs/sft_qwen3_8b_qlora.yaml \
  --base-model "${VERIFIED_MODEL}"
```

三张 3090 的本次配置、流水线和旧版 overnight 脚本已经隔离到
[`experiments/three-gpu-24gb/`](experiments/three-gpu-24gb/)，仅用于复现实验。

### 5. 亲自试用模型

安装项目后，使用一张 GPU 启动通用交互命令：

```bash
CUDA_VISIBLE_DEVICES=0 nuosu-chat \
  --model /absolute/path/to/verified-model \
  --adapter outputs/qwen3-8b-nuosu-nuosubench-sft-3gpu-fast \
  --tokenizer outputs/qwen3-8b-nuosu-nuosubench-sft-3gpu-fast
```

输入 `/clear` 清空多轮上下文，输入 `/quit` 退出。显存不足时可增加 `--load-in-4bit`。
单次测试使用：

```bash
nuosu-chat \
  --model /absolute/path/to/verified-model \
  --adapter /absolute/path/to/adapter \
  --prompt '请将“你好”翻译成凉山规范彝文。'
```

<img width="972" height="465" alt="image" src="https://github.com/user-attachments/assets/029221e3-7130-43a3-8f0d-4bc062e5d391" />

不懂彝文的测试者只能检查空输出、乱码、重复、截断、速度和任务格式，不能据此判断翻译
准确性。语义和规范性必须使用隐藏 reference 或母语者盲评。

## 标准工作流

```text
模型下载与 SHA 校验
        ↓
未训练底模生成冒烟
        ↓
64 条过拟合门禁
        ↓
CPT 或小规模 SFT
        ↓
validation 选择配置
        ↓
冻结配置
        ↓
保留测试集 + 母语者盲评
        ↓
发布 model card 与可复现清单
```

任一步失败都应停止，不用更多语料或更多 epoch 掩盖基础问题。

## 项目范围

第一阶段聚焦：

- 诺苏语北部方言；
- 圣乍话基础方言；
- 喜德标准音；
- 1980 年凉山规范彝文。

本项目不宣称覆盖全部彝语支系、传统彝文或所有地方口语。

## 数据来源

- [彝汉电子词典](https://www.yixueyanjiu.com/dict/)：词典翻译数据；
- [TianYeZ1214/NuosuBench](https://huggingface.co/datasets/TianYeZ1214/NuosuBench)：
  翻译/问答与保留研究测试；
- [isljsy/yidir](https://github.com/isljsy/yidir)：喜德读音研究与 PUA 映射参考；
- [NuosuBburma OCR Evaluation Set](https://huggingface.co/datasets/nanxidajun/NuosuBburma-OCR-Evaluation-Set)：
  连续文本 GT。

来源 revision、构建参数、方言标签和数据统计由 `nuosu-corpus` 管理。当前数据足以做研究
基线，但长文章、多轮对话和母语者评测仍明显不足。

## 仓库结构

```text
configs/              # CPT/SFT 配置；分级说明见 configs/README.md
docs/                 # 架构、训练、评测、部署和发布文档
evaluation/           # 评测协议与已归档报告
experiments/          # 与特定硬件和数据快照绑定的实验
patches/              # 按日期归档的一次性事故修复
scripts/
  model/              # 模型下载、校验和底模冒烟
  gates/              # 64 条过拟合与指标门槛
  *.py / *.sh         # 当前训练、评测和数据审计入口
src/nuosu_llm/        # 可测试的核心 Python 代码
tests/                # 单元测试
```

生成物统一写入被 Git 忽略的 `outputs/`、`artifacts/`、`logs/` 与
`evaluation/results/`。

## 当前实验记录

已通过底模校验、训练和正式评测的实验：

- [`evaluation/reports/2026-07-29-qwen3-8b-nuosu-formal.md`](evaluation/reports/2026-07-29-qwen3-8b-nuosu-formal.md)

损坏底模实验的结果只作为事故记录：

- [`evaluation/reports/2026-07-29-qwen3-8b-nuosu.md`](evaluation/reports/2026-07-29-qwen3-8b-nuosu.md)

不要上传该轮 adapter，也不要引用其分数作为模型能力。




## Development

```bash
make help
make test
make lint
```

贡献新的训练配置时，必须同时说明：基础模型本地路径与 revision、模型校验记录、数据
revision、随机种子、硬件、训练参数和 validation 结果。

## English

Nuosu LLM is a reproducible training and evaluation toolkit for Standard
Liangshan Yi (Nuosu). The first experimental run is invalid because all five
cached Base-model shards failed byte-level SHA-256 verification. The repository
now requires a quarantined download, checksum verification, a base-generation
smoke test, a 64-example overfit gate, and validation generation before a full
run.

Start with the generic single-GPU entry point:

```bash
python scripts/train.py \
  --config configs/sft_qwen3_8b_qlora.yaml \
  --base-model /absolute/path/to/verified-model
```

See [`docs/README.md`](docs/README.md) for the documentation map and
[`scripts/README.md`](scripts/README.md) for the command catalog. Hardware-specific
runs are archived under [`experiments/`](experiments/), while one-off incident
recovery tools live under [`patches/`](patches/).

To try a trained adapter interactively on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 nuosu-chat \
  --model /absolute/path/to/verified-model \
  --adapter /absolute/path/to/adapter
```

Use `/clear` to reset the conversation and `/quit` to exit. Non-Nuosu speakers
can test runtime behavior, formatting, repetition, and truncation, but semantic
and orthographic quality requires hidden references or blind native-speaker review.
