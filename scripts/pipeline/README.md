# Research pipeline / 研究训练流水线

```bash
bash scripts/pipeline/run_research_pipeline.sh /absolute/verified/model
```

执行顺序：

1. 64 条 SFT 过拟合门禁；
2. OCR GT CPT；
3. 词典 SFT；
4. NuosuBench 短样本 SFT；
5. NuosuBench 长尾 SFT。

任一阶段失败都会停止。流水线不会运行保留测试集，也不会发布模型。每阶段日志写入
`artifacts/pipeline/<UTC timestamp>/`。

服务器上已有后台恢复任务时，可条件等待：

```bash
bash scripts/pipeline/wait_for_recovery_then_train.sh \
  RECOVERY_PID \
  /absolute/verified/model \
  /absolute/recovery.log
```

只有恢复日志明确包含 `recovery passed` 且校验标记存在时才会启动训练。

如果 OCR CPT 和词典 SFT 已完成，而流水线在 NuosuBench 阶段中断，使用：

```bash
PYTHON_BIN=/absolute/path/to/python \
bash scripts/pipeline/resume_after_dictionary.sh /absolute/path/to/verified-model
```

恢复脚本要求词典 adapter 完整存在、short/long 输出尚不存在且 GPU 空闲。单机三卡
通信固定使用 loopback，并在 Accelerate 创建 barrier 前显式绑定 rank 与 CUDA 设备。
