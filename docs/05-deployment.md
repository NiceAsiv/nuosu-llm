# 部署指南

## 1. 部署目标

训练使用 GPU 服务器，线上服务使用较小的 INT4 模型。训练资源充足并不意味着线上必须部署大模型。

推荐首版：

- 模型：8B 主模型，必要时另行蒸馏或训练 4B 部署版；
- 量化：GGUF `Q4_K_M`；
- 上下文：先限制为 2K～4K；
- 并发：从 1 开始压测；
- 服务：`llama.cpp` 或 Ollama；
- 知识问答：使用 RAG，不把频繁变化的知识全部微调进模型。

## 2. 合并 adapter

```bash
python scripts/merge_adapter.py \
  --base-model outputs/qwen3-8b-cpt-merged \
  --adapter outputs/qwen3-8b-sft-qlora \
  --output artifacts/qwen3-8b-nuosu-merged
```

合并后先用 Transformers 做 BF16 回归测试，再转换 GGUF。

## 3. 转换 GGUF

安装与目标模型架构兼容的 `llama.cpp`：

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
python convert_hf_to_gguf.py \
  /path/to/qwen3-8b-nuosu-merged \
  --outfile /path/to/qwen3-8b-nuosu-f16.gguf \
  --outtype f16
```

量化：

```bash
./build/bin/llama-quantize \
  /path/to/qwen3-8b-nuosu-f16.gguf \
  /path/to/qwen3-8b-nuosu-q4_k_m.gguf \
  Q4_K_M
```

`llama.cpp` 的命令和模型支持会变化，转换前应使用与目标模型匹配的版本，并查看该版本文档。

## 4. 本地服务

```bash
./build/bin/llama-server \
  -m /path/to/qwen3-8b-nuosu-q4_k_m.gguf \
  -c 4096 \
  -ngl 99 \
  --host 0.0.0.0 \
  --port 8080
```

生产环境不要直接暴露到公网。应增加：

- 身份认证；
- 请求体和输出长度限制；
- 速率限制；
- 日志脱敏；
- 超时和熔断；
- 内容安全与人工反馈入口；
- 模型、adapter、量化版本记录。

## 5. RAG

适合放入 RAG 的内容：

- 教材和词典；
- 经授权的文化知识；
- 政策、办事和机构信息；
- 会持续更新的领域文档。

RAG 文档仍需保留来源和授权。回答中应返回引用，不确定时应明确说明，而不是生成看似流畅但无法核验的内容。

## 6. 压测指标

- 冷启动时间；
- 首 token 延迟；
- tokens/s；
- 峰值 RAM / VRAM；
- 2K、4K 上下文下的内存；
- 并发 1、2、4 时的 P50/P95；
- 超长输入和异常 Unicode；
- INT4 与 BF16 的语言质量差异。

## 7. 版本命名

建议：

```text
nuosu-qwen3-8b-cpt20m-sft10k-v0.1
nuosu-qwen3-8b-cpt20m-sft10k-q4km-v0.1
```

模型卡应记录：

- 基础模型和 revision；
- CPT / SFT 数据规模；
- 数据授权摘要；
- 评测结果；
- 已知限制；
- 支持的语言和书写体系；
- 不建议用途。
