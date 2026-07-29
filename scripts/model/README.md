# Verified model recovery / 模型恢复

`recover_qwen3_base.sh` 是训练前的第一个入口。

默认运行是只读 dry-run：

```bash
bash scripts/model/recover_qwen3_base.sh
```

实际执行：

```bash
PROXY_URL=http://127.0.0.1:7897 \
PYTHON_BIN=/absolute/path/to/python \
bash scripts/model/recover_qwen3_base.sh --execute
```

安全属性：

- 旧缓存与旧 adapter 只移动到带 UTC 时间戳的 quarantine，不删除；
- 下载固定到显式 revision；
- 下载使用独立 `HF_HOME`，不复用已损坏缓存；
- 五个 safetensors 分片必须通过实际 SHA-256；
- 底模必须通过三条可读生成检查；
- 脚本不会自动启动训练。

输出：

```text
../models/Qwen3-8B-Base-<revision>/VERIFIED.sha256
artifacts/model-recovery/<timestamp>/
../quarantine/<timestamp>/
```

若下载中断，保留目标目录以便排查。重新执行时，脚本会先把已有目标目录移入 quarantine，
不会在不确定状态下覆盖。
