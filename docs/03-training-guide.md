# 训练指南

## 1. 环境要求

- Linux；
- Python 3.10 或 3.11；
- NVIDIA 驱动可见；
- 单张 RTX 3090 24GB 或同等级 GPU；
- 建议预留 100GB 以上缓存和 checkpoint 空间。

宿主机不需要安装完整 CUDA Toolkit；PyTorch 或 Docker 镜像可以携带所需 CUDA runtime。驱动版本必须兼容容器中的 CUDA runtime。

## 2. 安装

```bash
git clone <repository-url> nuosu-llm
git clone https://github.com/NiceAsiv/nuosu-corpus.git
cd nuosu-llm
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[train,dev]"
```

检查：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import bitsandbytes; print(bitsandbytes.__version__)"
pytest -q
```

## 3. 阶段零：基线

在训练前保存以下结果：

- 候选 tokenizer 的 tokens/彝文字符；
- 固定提示集输出；
- 测试集困惑度或平均 loss；
- 母语者对准确性、自然度和语言纯度的评分；
- 推理显存、首 token 延迟和生成速度。

没有训练前基线，就无法判断微调是否带来真实提升。

## 4. CPT

CPT 使用纯文本 JSONL：

```bash
python scripts/train.py --config configs/cpt_qwen3_8b_qlora.yaml
```

默认参数适合单张 24GB GPU：

- 4-bit NF4；
- `target_modules: all-linear`；
- LoRA rank 32；
- 序列长度 2048；
- micro batch 1；
- 梯度累积 16；
- BF16；
- gradient checkpointing；
- 1 个 epoch。

首轮只训练 100万～500万 token，重点观察：

- eval loss 是否下降；
- 是否开始稳定生成规范彝文；
- 中文能力是否明显退化；
- 是否出现重复输出；
- 是否过拟合单一来源。

建议在 CPT 数据中保留 10%～30% 的高质量中文或彝汉双语文本，以减轻灾难性遗忘。具体比例应通过评测确定。

## 5. CPT adapter 处理

有两种路线：

### 路线 A：先合并 CPT，再做 SFT

优点是阶段边界清晰，便于保存语言模型检查点。

```bash
python scripts/merge_adapter.py \
  --base-model Qwen/Qwen3-8B-Base \
  --adapter outputs/qwen3-8b-cpt-qlora \
  --output outputs/qwen3-8b-cpt-merged
```

然后把 SFT 配置中的 `base_model` 改为：

```yaml
base_model: outputs/qwen3-8b-cpt-merged
```

### 路线 B：继续训练同一 adapter

存储开销较小，但实验追踪和最终合并更容易出错。第一版建议使用路线 A。

## 6. SFT

先在相邻的 `nuosu-corpus` 仓库完成 SFT 数据校验和构建，再运行：

```bash
python scripts/train.py \
  --config configs/sft_qwen3_8b_qlora.yaml
```

SFT 使用 `assistant_only_loss: true`，只在 assistant 输出部分计算损失。若模型的 chat template 不支持 generation mask，应停止训练并检查模板，不能默默退化为全序列 loss。

## 7. 多卡

8B QLoRA 可在单张 24GB RTX 3090 上以 2K 上下文、micro batch 1 起步。单卡通常最稳定；只有吞吐量或更大模型确实需要时再使用多卡：

```bash
accelerate config
accelerate launch --num_processes 3 \
  scripts/train.py \
  --config configs/cpt_qwen3_8b_qlora.yaml
```

三张 RTX 3090 若没有 NVLink，跨卡通信走 PCIe。多卡可以提高吞吐量，但不会自动把有效 batch、学习率和保存策略调到合理值。

## 8. 断点恢复

配置：

```yaml
training:
  resume_from_checkpoint: outputs/qwen3-8b-cpt-qlora/checkpoint-1000
```

恢复前确认：

- 基础模型、数据版本、配置未变化；
- adapter rank 和 target modules 一致；
- checkpoint 完整；
- Git commit 和数据清单已记录。

## 9. 实验记录

每次训练至少记录：

```yaml
run_id:
git_commit:
base_model:
base_model_revision:
dataset_manifest:
train_tokens:
eval_tokens:
random_seed:
config_file:
gpu_model:
library_versions:
start_time:
end_time:
result_summary:
```

不要只保留“最终效果最好”的 checkpoint。失败实验同样能避免后续重复踩坑。

## 10. 常见问题

### loss 下降但生成仍是乱码

优先检查 tokenizer、解码、Unicode 规范化和数据编码；不要直接增加 epoch。

### 模型只会翻译

重新平衡 SFT 任务分布，加入诺苏语问答、改写、多轮对话和拒答样本。

### 中文能力严重退化

降低 CPT 学习率或 epoch，并增加高质量中文/双语保持数据。

### 输出重复

检查训练数据重复率、长度分布和过拟合；推理时再谨慎调整 repetition/presence penalty，不能只靠采样参数掩盖训练问题。
