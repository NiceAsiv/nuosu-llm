# Verified model acquisition / 模型获取与验证

稳定目录只提供可组合的模型工具：

- `download_snapshot.py`：下载固定仓库 revision；
- `manifests/`：保存已审核 revision 的实际 SHA-256；
- `smoke_test_base.py`：检查未训练底模的基本生成。

```bash
MODEL_DIR=/absolute/path/to/models/Qwen3-8B-Base-49e3418fbbbc
python scripts/model/download_snapshot.py \
  --repo-id Qwen/Qwen3-8B-Base \
  --revision 49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
  --output-dir "${MODEL_DIR}"
```

```bash
(cd "${MODEL_DIR}" && \
  sha256sum --check /absolute/path/to/nuosu-llm/scripts/model/manifests/qwen3-8b-base-49e3418.sha256)
python scripts/model/smoke_test_base.py --model "${MODEL_DIR}"
cp scripts/model/manifests/qwen3-8b-base-49e3418.sha256 "${MODEL_DIR}/VERIFIED.sha256"
```

For the pinned 1.7B dictionary model candidate:

```bash
MODEL_DIR=/absolute/path/to/models/Qwen3-1.7B-Base-36be17a0
python scripts/model/download_snapshot.py \
  --repo-id Qwen/Qwen3-1.7B-Base \
  --revision 36be17a0ee54955c2d50eb4af5a126b429874a6e \
  --output-dir "${MODEL_DIR}"
(cd "${MODEL_DIR}" && \
  sha256sum --check /absolute/path/to/nuosu-llm/scripts/model/manifests/qwen3-1.7b-base-36be17a.sha256)
python scripts/model/smoke_test_base.py --model "${MODEL_DIR}"
cp scripts/model/manifests/qwen3-1.7b-base-36be17a.sha256 "${MODEL_DIR}/VERIFIED.sha256"
```

训练入口要求本地模型目录存在 `VERIFIED.sha256`。如果需要隔离已经损坏的缓存和派生
adapter，请使用按日期归档的 [`../../patches/2026-07-29/`](../../patches/2026-07-29/)；
该补丁不是正常下载流程。
