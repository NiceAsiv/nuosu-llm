#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_DIR="${1:?usage: resume_after_dictionary.sh /absolute/verified/model}"
DICTIONARY_ADAPTER="${PROJECT_DIR}/outputs/qwen3-8b-nuosu-dictionary-sft-3gpu-fast"
SHORT_OUTPUT="${PROJECT_DIR}/outputs/qwen3-8b-nuosu-nuosubench-short-sft-3gpu-fast"
LONG_OUTPUT="${PROJECT_DIR}/outputs/qwen3-8b-nuosu-nuosubench-sft-3gpu-fast"
RUN_ID="$(date -u "+%Y%m%dT%H%M%SZ")"
LOG_DIR="${PROJECT_DIR}/artifacts/pipeline-resume/${RUN_ID}"

[[ -f "${MODEL_DIR}/VERIFIED.sha256" ]] || {
  echo "verified model marker missing: ${MODEL_DIR}/VERIFIED.sha256" >&2
  exit 2
}
[[ -f "${DICTIONARY_ADAPTER}/adapter_model.safetensors" ]] || {
  echo "completed dictionary adapter missing: ${DICTIONARY_ADAPTER}" >&2
  exit 2
}
[[ ! -e "${SHORT_OUTPUT}" ]] || {
  echo "short-stage output already exists; isolate it before resuming: ${SHORT_OUTPUT}" >&2
  exit 2
}
[[ ! -e "${LONG_OUTPUT}" ]] || {
  echo "long-stage output already exists; isolate it before resuming: ${LONG_OUTPUT}" >&2
  exit 2
}
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU compute processes are active; refusing to overlap the resume run." >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

run_stage() {
  local stage_name="$1"
  local config_path="$2"

  echo "[$(date -u "+%Y-%m-%dT%H:%M:%SZ")] starting ${stage_name}"
  env \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES=0,1,2 \
    NCCL_SOCKET_IFNAME=lo \
    GLOO_SOCKET_IFNAME=lo \
    NCCL_IB_DISABLE=1 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" -m torch.distributed.run \
      --standalone \
      --nproc_per_node=3 \
      scripts/train.py \
      --config "${config_path}" \
      --base-model "${MODEL_DIR}" 2>&1 | tee "${LOG_DIR}/${stage_name}.log"
}

run_stage \
  sft-nuosubench-short \
  experiments/xjtu-3gpu/configs/sft_qwen3_8b_nuosubench_research_3gpu_fast.yaml
run_stage \
  sft-nuosubench-long \
  experiments/xjtu-3gpu/configs/sft_qwen3_8b_nuosubench_long_research_3gpu_fast.yaml

echo "resume pipeline completed; run validation generation before held-out testing"
