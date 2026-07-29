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
