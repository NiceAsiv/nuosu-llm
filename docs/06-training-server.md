# 训练服务器建议

## 1. 已核验硬件

目标训练服务器具备：

- 2 × Intel Xeon Platinum 8163；
- 48 核 / 96 线程；
- 约 1TB 内存；
- 3 × NVIDIA RTX 3090 24GB；
- NVIDIA 驱动支持 CUDA 12.8；
- NVMe 可用空间约 1.4TB；
- `/data0` 可用空间约 9.9TB；
- Docker 已安装；
- GPU 当前可用。

三张 GPU 通过 PCIe 互联，未检测到 NVLink。默认 8B QLoRA 优先使用单卡，减少多卡通信和配置复杂度；其余 GPU 可并行运行基线或超参数实验。

## 2. 当前软件状态

- Ubuntu 20.04；
- 系统 Python 3.8；
- 默认 Python 未安装 PyTorch；
- 未发现 `nvcc`；
- Docker 可用。

Python 3.8 已不适合作为本项目默认训练环境。推荐使用 Docker，或者在用户目录安装独立 Python 3.10/3.11 环境，不修改系统 Python。

## 3. 推荐目录

```text
/data0/nuosu/
├── cache/huggingface/
├── nuosu-corpus/
│   ├── raw/
│   └── processed/
├── checkpoints/
└── artifacts/
```

项目代码可以位于用户主目录，模型缓存、数据和 checkpoint 放到 `/data0`。

环境变量：

```bash
export HF_HOME=/data0/nuosu/cache/huggingface
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true
```

## 4. Docker 启动示例

从项目根目录执行：

```bash
docker build -f docker/Dockerfile -t nuosu-llm:dev .

docker run --rm -it \
  --gpus '"device=0"' \
  --ipc=host \
  --shm-size=32g \
  -v "$PWD:/workspace/nuosu-llm" \
  -v "../nuosu-corpus:/workspace/nuosu-corpus" \
  -v /data0/nuosu:/data0/nuosu \
  -e HF_HOME=/data0/nuosu/cache/huggingface \
  nuosu-llm:dev
```

首次运行前验证：

```bash
nvidia-smi
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python3 -m pytest -q
```

## 5. 单卡预算

经验起点：

| 模型 | 方法 | 单卡 24GB |
|---|---|---|
| 0.6B～2B | QLoRA | 充足 |
| 3B～4B | QLoRA | 充足 |
| 7B～8B | QLoRA | 可行，需控制上下文与 batch |
| 14B | QLoRA | 单卡偏紧，需多卡或进一步优化 |

本项目默认以 Qwen3-8B-Base 为主模型。先用少量语料跑通 8B 的 CPT、SFT 和 NuosuBench 评测，再扩大数据规模。

## 6. 训练前检查

```bash
nvidia-smi
df -h /data0
git status
(cd ../nuosu-corpus && python scripts/validate_dataset.py --stage cpt --input <train.jsonl>)
python scripts/audit_tokenizers.py --input <audit.txt> --models <model>
```

不要在没有固定评测集、数据清单和可恢复 checkpoint 策略的情况下启动长时间训练。
