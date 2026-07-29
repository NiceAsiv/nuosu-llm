# Legacy experiment automation

这些脚本用于复现早期服务器实验，不是推荐入口：

- `run_overnight_pipeline.sh`
- `run_long_tail_when_ready.sh`

它们假定固定服务器路径、已有 adapter 和特定三卡环境。新的训练必须先运行
`../model/recover_qwen3_base.sh`，并逐阶段执行门禁。
