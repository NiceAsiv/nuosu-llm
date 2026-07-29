# Experiments / 实验归档

这里保存与特定硬件、数据快照或研究阶段绑定的配置和编排脚本。它们用于复现实验，不是
项目的默认入口。

通用用户应从根目录 `README.md` 和 `configs/*_qlora.yaml` 开始。需要多卡时，用实际可用
GPU 数量启动相同的训练入口：

```bash
torchrun --standalone --nproc_per_node=NUM_GPUS \
  scripts/train.py \
  --config configs/sft_qwen3_8b_qlora.yaml \
  --base-model /absolute/path/to/verified-model
```

复制实验配置前必须重新核对：

- GPU 型号、数量、显存和互联方式；
- 有效全局 batch；
- 数据长度分布；
- PyTorch、Transformers、TRL 和 bitsandbytes 版本；
- attention backend、精度与吞吐；
- 输出目录和前序 adapter。

当前归档：

- [`three-gpu-24gb/`](three-gpu-24gb/)：三张 24GB GPU 的 2026-07-29 研究运行。
