# Training gates / 训练门禁

`run_sft_overfit_64.sh` 从词典训练集确定性提取 64 条，在干净 Base 上训练一个临时 LoRA，
再对同一批样本生成并检查：

- compact exact match ≥ 0.50；
- chrF2 ≥ 70；
- Unicode 替换字符率 ≤ 0.01；
- 长度截断率 ≤ 0.05。

门禁用于证明训练、assistant mask、adapter 加载、EOS 和解码链路能够工作，不代表泛化能力。
门禁失败时禁止启动完整训练。
