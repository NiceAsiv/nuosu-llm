# Three-GPU 24GB experiment / 三卡 24GB 实验

这是特定服务器实验的可复现快照，不是所有用户都需要三张 GPU。

## 环境

- 3 × NVIDIA RTX 3090 24GB；
- 单机数据并行；
- GPU `0,1,2`；
- Qwen3-8B-Base 固定 revision；
- QLoRA、BF16/FP32、SDPA；
- `nuosu-corpus` 与本仓库位于同一父目录。

`run_pipeline.sh` 中的 loopback、禁用 InfiniBand、batch size 和输出目录都是这次服务器
运行的选择。其他机器不应直接照搬。

## 复现实验

```bash
PYTHON_BIN=/absolute/path/to/python \
bash experiments/three-gpu-24gb/run_pipeline.sh /absolute/path/to/verified-model
```

流水线依次执行：

1. 64 条 SFT 过拟合门禁；
2. OCR GT CPT；
3. 词典 SFT；
4. NuosuBench 短样本 SFT；
5. NuosuBench 长样本 SFT。

`configs/` 保存本次实验的确切参数。`legacy/` 是更早的 watcher/overnight 自动化，仅供追溯。
事故中断后的恢复脚本位于 [`../../patches/2026-07-29/`](../../patches/2026-07-29/)。

## 移植到其他硬件

不要修改本目录来制造“通用配置”。请复制根目录的 QLoRA 模板并重新做数据画像和吞吐
测试。GPU 数量由 `torchrun --nproc_per_node` 决定，配置中的
`per_device_train_batch_size × gradient_accumulation_steps × world_size` 决定有效全局
batch。
