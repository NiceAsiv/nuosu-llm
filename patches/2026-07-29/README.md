# 2026-07-29 recovery patches

这些脚本只用于恢复本项目 2026-07-29 的服务器实验：

| 文件 | 用途 |
|---|---|
| `recover_corrupt_qwen3_base.sh` | 隔离损坏缓存及其派生 adapter，重新下载固定 revision，校验 SHA 并做底模冒烟 |
| `wait_for_recovery_then_train.sh` | 等待上述恢复完成，再启动 XJTU 三卡实验 |
| `resume_after_dictionary.sh` | OCR CPT 和词典 SFT 已完成时，只恢复短样本和长样本阶段 |

它们包含这次运行的模型 revision、输出命名、三卡环境和阶段依赖。新用户不需要运行这些
补丁，除非遇到相同事故并已逐项确认前置条件。

恢复脚本默认 dry-run；`--execute` 才会移动缓存和旧输出。移动目标进入带时间戳的
`quarantine/`，不会直接删除。

完整复盘见 [`../../docs/07-lessons-learned.md`](../../docs/07-lessons-learned.md)。
