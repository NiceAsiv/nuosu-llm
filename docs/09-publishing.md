# 代码、数据与模型发布

## 1. 推荐结论

代码仓库放 GitHub；数据集和模型权重放 Hugging Face。两者互相链接，不要二选一。

| 内容 | 推荐平台 | 原因 |
|---|---|---|
| Python 代码、配置、文档、Issue | GitHub | 代码审查、CI、版本管理和协作更成熟 |
| 训练语料 | Hugging Face Dataset | Dataset Viewer、数据卡、版本和 `load_dataset` |
| LoRA adapter、合并模型、GGUF | Hugging Face Model | 大文件托管、模型卡和推理生态 |
| NuosuBench | 引用原作者仓库 | 避免不必要的镜像和版本分叉 |
| 实验日志和私有原始数据 | 私有存储 | 可能包含授权、隐私或敏感内容 |

## 2. 建议命名

GitHub：

```text
<organization>/nuosu-corpus
<organization>/nuosu-llm
```

Hugging Face：

```text
<organization>/nuosu-corpus-v1
<organization>/qwen3-8b-nuosu-cpt
<organization>/qwen3-8b-nuosu-instruct
<organization>/qwen3-8b-nuosu-gguf
```

如果尚未成立组织，可先使用个人命名空间，后续再转移到组织。

## 3. GitHub 应包含

- `src/`、`scripts/`、`configs/`、`tests/`；
- 所有 Markdown 文档；
- 小型、明确为虚构的格式样例；
- CI；
- `LICENSE`；
- 数据和模型的获取说明；
- Hugging Face 模型/数据链接；
- 复现实验所需的 commit 和版本信息。

两个 GitHub 仓库都不应包含：

- 模型权重；
- Hugging Face 缓存；
- 原始授权语料；
- NuosuBench 的重复副本；
- token、SSH 信息和内网地址；
- 评审者个人信息。

## 4. Hugging Face Dataset 发布

发布前确认许可证允许“再分发”，而不仅仅是允许内部训练。

数据卡至少记录：

- 语言：`ii` / `iii`，不能写成 `yi`；
- 方言和书写体系；
- 数据来源与授权；
- train/validation/test 划分；
- OCR、机器翻译和母语者校对比例；
- 个人信息处理；
- 已知偏差；
- 与 NuosuBench 的去重方式。

如果原始文本不能公开，可以只发布：

- 处理脚本；
- 数据清单和统计；
- 可公开子集；
- 数据申请流程；
- 使用私有/gated Hugging Face Dataset。

## 5. Hugging Face Model 发布

每个模型仓库记录：

- 基础模型和精确 revision；
- 对应 GitHub commit；
- CPT/SFT 数据版本和规模；
- QLoRA 参数；
- NuosuBench 和人工评测结果；
- 是否接触过 benchmark；
- 支持范围和不支持范围；
- 量化方式；
- 许可证和基础模型要求。

LoRA adapter 与合并模型最好分开发布，避免使用者误解。

## 6. 版本关联

一次正式发布应包含：

```text
GitHub tag:             v0.1.0
Git commit:             <sha>
Dataset revision:       <sha>
Base model revision:    <sha>
Adapter revision:       <sha>
Merged model revision:  <sha>
Evaluation manifest:    <sha>
```

模型卡中的训练命令必须指向发布 tag，而不是不断变化的 `main`。

## 7. 推荐工作流

1. 先在 GitHub 建立私有仓库；
2. 配置 CI，确保测试通过；
3. 清理密钥、内网地址和数据文件；
4. 完成第一轮 tokenizer/CPT 冒烟实验；
5. 确认数据授权后创建 Hugging Face Dataset；
6. 发布 LoRA adapter 和模型卡；
7. 评测通过后再公开合并模型和 GGUF；
8. GitHub Release 与 Hugging Face revision 互相引用。
