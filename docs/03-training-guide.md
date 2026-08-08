# 训练指南

## 1. 环境要求

- Linux；
- Python 3.10 或 3.11；
- NVIDIA 驱动可见；
- 单张 RTX 3090 24GB 或同等级 GPU；
- 建议预留 100GB 以上缓存和 checkpoint 空间。

## 0. 推荐的一条命令 MT 流程

正式的 Qwen3-1.7B 翻译实验不要求人工逐段启动：

```bash
NUM_GPUS=3 bash recipes/qwen3-1.7b-mt/run.sh
```

它依次完成 MT 数据投影与拒绝审计、1,165 个音节和55个部首的词表扩充、CPT、SFT、
断点恢复、基础模型与最终 adapter 的生成式评测、chrF2/CER/精确匹配评分、数值检查和哈希
清单。只有 `COMPLETED` 标记存在时才能登记为完成实验；`FAILED` 会记录失败时间和退出码。

新增 token 不是随机初始化：每个 token 使用扩充前子词 embedding 的均值初始化。PEFT
只开放新增 token 行以及 LoRA 参数，避免训练和保存完整的15万词 embedding 矩阵。

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

先按 [`../scripts/model/README.md`](../scripts/model/README.md) 下载固定 revision、校验每个
权重分片，并完成未训练底模冒烟：

```bash
export VERIFIED_MODEL=/absolute/path/to/verified/model
test -f "${VERIFIED_MODEL}/VERIFIED.sha256"
```

训练入口默认拒绝 Hugging Face Model ID 和缺少 `VERIFIED.sha256` 的目录。这样可以避免缓存
链接名正确、实际大文件内容损坏时继续训练。

已经发生坏缓存事故时，再使用 [`../patches/`](../patches/) 中与事故日期对应的补丁。补丁
不是正常训练的前置步骤。

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
python scripts/train.py \
  --config configs/cpt_qwen3_8b_qlora.yaml \
  --base-model "${VERIFIED_MODEL}"
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
  --base-model "${VERIFIED_MODEL}" \
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
  --config configs/sft_qwen3_8b_qlora.yaml \
  --base-model "${VERIFIED_MODEL}"
```

SFT 使用 `assistant_only_loss: true`，只在 assistant 输出部分计算损失。若模型的 chat template 不支持 generation mask，应停止训练并检查模板，不能默默退化为全序列 loss。

本仓库会在 Qwen3 Base 模板缺少 `{% generation %}` 区段时安装兼容的 ChatML 模板，并要求
tokenizer 返回 assistant mask。更换基础模型时仍应先做短基准，确认 assistant token 数量非零。

## 7. 多卡

8B QLoRA 可在单张 24GB RTX 3090 上以 2K 上下文、micro batch 1 起步。单卡通常最稳定；只有吞吐量或更大模型确实需要时再使用多卡：

```bash
NUM_GPUS=2
torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
  scripts/train.py \
  --config configs/cpt_qwen3_8b_qlora.yaml \
  --base-model "${VERIFIED_MODEL}"
```

`NUM_GPUS` 可以是 2、3、4 或当前机器实际可用的数量。多卡可以提高吞吐量，但不会自动
把有效 batch、学习率和保存策略调到合理值。

数据并行的有效全局 batch 为：

```text
world_size × per_device_train_batch_size × gradient_accumulation_steps
```

短样本应优先增加 `per_device_train_batch_size`，让一次前后向覆盖更多 token，再减少梯度累积。
不要让每张卡连续跑多个 batch=1 的微批次后才同步。推荐流程：

```bash
python scripts/profile_dataset.py \
  --config configs/sft_qwen3_8b_qlora.yaml \
  --batch-size 32

bash scripts/benchmark_throughput.sh \
  configs/sft_qwen3_8b_qlora.yaml \
  30 \
  outputs/previous-adapter-or-checkpoint
```

`benchmark_throughput.sh` 同样要求先设置 `BASE_MODEL="${VERIFIED_MODEL}"`。

基准至少检查：

- 最大显存保留 1GB 以上余量；
- 三张卡稳定阶段利用率，而不是模型加载时的瞬时值；
- `train_samples_per_second` 和 `num_tokens`；
- loss、gradient norm 是否有限；
- 第一轮 eval 是否成功。

对于长度长尾明显的数据，启用 `group_by_length`，并根据 P99 而不是最大值选择 batch。确需保留
极少数超长样本时，可把它们拆成独立的长上下文阶段，避免所有短样本都使用 batch=1。

本仓库提供可复现的 token 长度分桶：

```bash
python scripts/bucket_sft_by_length.py \
  --config configs/sft_qwen3_8b_qlora.yaml \
  --threshold 512 \
  --output-dir artifacts/ready_sft_length_buckets
```

短阶段和长阶段必须沿用同一 adapter；不能把两个阶段各自从基础模型开始后再尝试合并。
本项目三张 RTX 3090 的确切参数和分阶段流水线只在
[`../experiments/three-gpu-24gb/`](../experiments/three-gpu-24gb/) 中归档。

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

第一步对每个 safetensors 计算实际 SHA-256，不能只看 Hugging Face 缓存链接名。然后检查
未训练底模、tokenizer 往返编码、解码和 Unicode 规范化。底模冒烟失败时，loss 曲线无论
多平滑都不能证明训练有效，也不要直接增加 epoch。

### 模型只会翻译

重新平衡 SFT 任务分布，加入诺苏语问答、改写、多轮对话和拒答样本。

### 中文能力严重退化

降低 CPT 学习率或 epoch，并增加高质量中文/双语保持数据。

### 输出重复

检查训练数据重复率、长度分布和过拟合；推理时再谨慎调整 repetition/presence penalty，不能只靠采样参数掩盖训练问题。
