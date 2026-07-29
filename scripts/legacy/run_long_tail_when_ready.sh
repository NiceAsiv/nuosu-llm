#!/usr/bin/env bash
# Historical watcher retained for experiment reproduction.
# It is not a supported entry point for a new training run.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xjtuoss/nuosu-project/nuosu-llm}"
PYTHON_BIN="${PYTHON_BIN:-/home/xjtuoss/nuosu-project/.venv/bin/python}"
SHORT_ADAPTER="${PROJECT_DIR}/outputs/qwen3-8b-nuosu-nuosubench-short-sft-3gpu-fast/adapter_model.safetensors"
FINAL_ADAPTER="${PROJECT_DIR}/outputs/qwen3-8b-nuosu-nuosubench-sft-3gpu-fast/adapter_model.safetensors"
LONG_CONFIG="configs/sft_qwen3_8b_nuosubench_long_research_3gpu_fast.yaml"
LOG_PATH="${PROJECT_DIR}/logs/sft_nuosubench_long_3gpu.log"

cd "${PROJECT_DIR}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

if [[ -f "${FINAL_ADAPTER}" ]]; then
  echo "$(timestamp) final long-tail adapter already exists"
  exit 0
fi

echo "$(timestamp) waiting for short-stage adapter"
while [[ ! -f "${SHORT_ADAPTER}" ]]; do
  sleep 30
done

# Give the parent pipeline time to launch its own long-tail stage. If it does,
# this watcher observes the active trainer and exits after the final adapter is saved.
idle_checks=0
while (( idle_checks < 3 )); do
  if [[ -f "${FINAL_ADAPTER}" ]]; then
    echo "$(timestamp) final long-tail adapter completed by parent pipeline"
    exit 0
  fi
  if pgrep -f 'scripts/train.py' >/dev/null; then
    idle_checks=0
  else
    idle_checks=$((idle_checks + 1))
  fi
  sleep 10
done

echo "$(timestamp) running five-step long-context peak-memory benchmark"
bash scripts/benchmark_throughput.sh \
  "${LONG_CONFIG}" \
  5 \
  "$(dirname "${SHORT_ADAPTER}")"

if [[ -f "${FINAL_ADAPTER}" ]]; then
  echo "$(timestamp) final adapter appeared during benchmark"
  exit 0
fi

echo "$(timestamp) starting long-tail training"
env \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2 \
  OMP_NUM_THREADS=1 \
  PYTHONUNBUFFERED=1 \
  "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=3 \
    scripts/train.py \
    --config "${LONG_CONFIG}" 2>&1 | tee "${LOG_PATH}"
echo "$(timestamp) completed long-tail training"
