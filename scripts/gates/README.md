# Training gates / 训练门禁

`run_sft_overfit_64.sh` 从统一 SFT 训练集确定性提取前 64 条规范字—拼音—IPA 记录，
在干净的 post-trained Base 上训练一个临时 LoRA，
再对同一批样本生成并检查：

- compact exact match ≥ 0.90；
- chrF2 ≥ 90；
- Unicode 替换字符率 ≤ 0.01；
- 长度截断率 ≤ 0.05。

门禁用于证明训练、assistant mask、adapter 加载、EOS 和解码链路能够工作，不代表泛化能力。
门禁失败时禁止启动完整训练。

门禁与正式 SFT 使用同一条 prompt/completion 路径、官方 Qwen3 chat template 和上下文
`/no_think`。这样门禁覆盖实际训练的监督掩码、EOS 和解码链路，避免只验证另一套模板。
