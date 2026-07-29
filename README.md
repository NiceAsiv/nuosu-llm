# Nuosu LLM

[![CI](https://github.com/NiceAsiv/nuosu-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/NiceAsiv/nuosu-llm/actions/workflows/ci.yml)

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

### 2. 恢复并验证基础模型

先看执行计划，不修改任何文件：

```bash
bash scripts/model/recover_qwen3_base.sh
```

确认路径后执行：

```bash
PROXY_URL=http://127.0.0.1:7897 \
PYTHON_BIN="$(pwd)/.venv/bin/python" \
bash scripts/model/recover_qwen3_base.sh --execute
```

该脚本只移动隔离，不删除数据，并严格按顺序执行：

1. 隔离损坏的 Hugging Face 缓存；
2. 隔离由坏底模训练出的 `outputs/qwen3-8b-*`；
3. 下载固定 revision 到独立目录；
4. 对五个权重分片计算实际 SHA-256；
5. 用中文、英文和算术续写检查底模；
6. 通过后写入 `VERIFIED.sha256`，但**不会自动开始训练**。

### 3. 检查语料

```bash
python scripts/profile_dataset.py \
  --config configs/sft_qwen3_8b_dictionary_research_3gpu_fast.yaml \
  --batch-size 32
```

语料必须先由 `nuosu-corpus` 生成并校验。训练仓库不负责网页采集，也不应直接修改
`data/processed`。

### 4. 只在门禁通过后训练

恢复脚本会打印经过校验的本地模型目录。训练时显式传入该路径：

```bash
python scripts/train.py \
  --config configs/cpt_qwen3_8b_ocr_gt_research_3gpu.yaml \
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

运行门禁并在通过后依次启动 OCR CPT、词典 SFT、NuosuBench 短样本和长尾阶段：

```bash
bash scripts/pipeline/run_research_pipeline.sh "${VERIFIED_MODEL}"
```

任一阶段失败都会终止流水线。该流水线不会自动运行保留测试集。

当前仓库不会把历史整夜训练脚本当作推荐入口；它们保留在 `scripts/legacy/`，仅用于复现实验。

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
scripts/
  model/              # 模型恢复、校验和底模门禁
  gates/              # 64 条过拟合与指标门槛
  pipeline/           # 通过门禁后的分阶段训练
  legacy/             # 非推荐的历史自动化流程
  *.py / *.sh         # 当前训练、评测和数据审计入口
src/nuosu_llm/        # 可测试的核心 Python 代码
tests/                # 单元测试
```

生成物统一写入被 Git 忽略的 `outputs/`、`artifacts/`、`logs/` 与
`evaluation/results/`。

## 当前实验记录

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

Start with:

```bash
bash scripts/model/recover_qwen3_base.sh
bash scripts/model/recover_qwen3_base.sh --execute
```

See [`docs/README.md`](docs/README.md) for the documentation map and
[`scripts/README.md`](scripts/README.md) for the command catalog.
