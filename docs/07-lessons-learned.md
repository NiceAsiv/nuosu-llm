# 失败复盘与经验教训

本文记录 2026-07-29 第一轮 Qwen3-8B-Base 诺苏语适配实验。目的不是保留一串临时命令，
而是说明失败如何被发现、哪些结果作废、哪些保护已经进入稳定代码。

## 结论

第一轮 adapter 和评测结果全部作废，原因是本地 Qwen3-8B-Base 五个权重分片的实际
SHA-256 均与固定 revision 不一致。文件名和 Hugging Face 缓存链接看似正常，不能证明
权重内容正确。

重新下载并通过五个分片校验、底模生成冒烟和 SFT 小样本门禁后，才开始第二轮训练。

## 事故一：损坏底模仍能加载和训练

### 现象

- Transformers 可以加载模型；
- loss 能正常输出并下降；
- 生成结果却出现乱码和异常行为；
- 仅检查缓存目录、文件大小和符号链接没有发现问题。

### 根因

实际 safetensors 内容与固定 revision 的 SHA-256 不一致。模型“能加载”不是完整性验证。

### 处理

- 隔离坏缓存，不在原目录覆盖；
- 隔离由坏底模产生的所有 adapter；
- 固定仓库 revision；
- 对每个权重分片执行实际 SHA-256 校验；
- 用中文、英文和简单推理提示对未训练底模做生成冒烟；
- 只有通过后才写入 `VERIFIED.sha256`。

### 固化规则

训练入口拒绝远程 Model ID 和缺少校验标记的本地模型目录。事故处置脚本保留在
`patches/2026-07-29/`，正常下载、校验和冒烟能力保留在 `scripts/model/`。

## 事故二：Qwen3 Base 无 assistant mask

### 现象

SFT 开启 `assistant_only_loss` 后，TRL 无法从 Qwen3 Base 的聊天模板生成 assistant mask。

### 根因

基础模型模板没有 TRL 所需的 `{% generation %}` 区段。静默退化为全序列 loss 会改变
训练目标，因此不能接受。

### 处理

当前项目模型范围固定为 Qwen3 Base：训练入口在 `assistant_only_loss` 开启且模板缺少
generation 区段时安装兼容 ChatML 模板，并验证 assistant token 数量非零。将来支持其他
模型家族前，必须先按模型类型分派模板，不能把 Qwen 模板当作通用回退。

## 事故三：三卡流水线在数据预处理屏障中断

### 现象

- rank 0 完成约 8.8 万条样本的 tokenization 和 truncation；
- 其他 rank 在进程屏障附近退出；
- 日志出现 NCCL `connection reset by peer`；
- 三张 GPU 随后空闲；
- 系统内存、显存、磁盘和内核日志均没有 OOM 证据。

### 判断

已确认失败发生在单机多进程通信/设备绑定附近；无法仅凭一次日志证明唯一根因。显式绑定
rank 与 CUDA 设备，并在该服务器实验中固定 loopback、禁用不存在的 InfiniBand 后，
恢复训练成功越过屏障并由三张卡共同训练。

loopback 和 `NCCL_IB_DISABLE=1` 是这台服务器的实验参数，不是项目默认值。因此它们只
保留在 `experiments/three-gpu-24gb/` 和对应恢复补丁中。显式 rank/device 绑定属于通用正确性
保护，保留在核心训练代码。

## 事故四：脚本和配置开始掩盖主入口

### 现象

为了追赶服务器状态，仓库逐渐出现三卡、fast、overnight、wait、resume 等多个入口。它们
对当前机器有用，但新用户很难判断自己应该运行哪个，也容易误以为三卡是项目要求。

### 改进

仓库按职责重新分层：

```text
configs/                 通用单卡起点和门禁配置
scripts/                 稳定、可测试的训练与评测工具
experiments/three-gpu-24gb/   特定三卡服务器的配置和流水线
patches/2026-07-29/      一次性事故恢复脚本
evaluation/reports/      评测结果和已作废实验记录
```

正常用户只需一个训练入口：单卡直接运行 `scripts/train.py`，多卡用自己的 GPU 数量通过
`torchrun --nproc_per_node=N` 启动同一入口。GPU 数量不是数据集或模型的固有要求。

## 仍需遵守的实验原则

- loss 下降只能说明优化过程在工作，不能单独证明语言质量；
- 基础模型、数据 revision、代码 commit 和配置必须一起记录；
- 训练集、validation 和保留测试集必须物理隔离并做泄漏检查；
- 每个阶段先做小样本门禁，再扩大数据量；
- 硬件专用优化必须进入 `experiments/`，不能冒充默认值；
- 失败输出不发布，失败原因和复现证据要保留。

第一轮无效结果的具体评测记录见
[`../evaluation/reports/2026-07-29-qwen3-8b-nuosu.md`](../evaluation/reports/2026-07-29-qwen3-8b-nuosu.md)。
