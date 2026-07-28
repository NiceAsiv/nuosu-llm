#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xjtuoss/nuosu-project/nuosu-llm}"
PYTHON_BIN="${PYTHON_BIN:-/home/xjtuoss/nuosu-project/.venv/bin/python}"
WAIT_PID="${1:-}"
DICTIONARY_INIT_ADAPTER="${DICTIONARY_INIT_ADAPTER:-}"
LOG_DIR="${PROJECT_DIR}/logs"
CPT_ADAPTER="${PROJECT_DIR}/outputs/qwen3-8b-nuosu-ocr-gt-cpt-3gpu-stable/adapter_model.safetensors"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

if [[ -n "${WAIT_PID}" ]]; then
  echo "$(timestamp) waiting for current CPT PID ${WAIT_PID}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 30
  done
fi

if [[ ! -f "${CPT_ADAPTER}" ]]; then
  echo "$(timestamp) CPT adapter missing: ${CPT_ADAPTER}" >&2
  exit 1
fi

run_stage() {
  local stage_name="$1"
  local config_path="$2"
  local init_adapter="${3:-}"
  local stage_log="${LOG_DIR}/${stage_name}.log"
  local command=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --standalone
    --nproc_per_node=3
    scripts/train.py
    --config "${config_path}"
  )

  if [[ -n "${init_adapter}" ]]; then
    command+=(--init-adapter "${init_adapter}")
  fi

  echo "$(timestamp) starting ${stage_name}: ${config_path}"
  env \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES=0,1,2 \
    OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    "${command[@]}" 2>&1 | tee "${stage_log}"
  echo "$(timestamp) completed ${stage_name}"
}

run_stage \
  "sft_dictionary_3gpu" \
  "configs/sft_qwen3_8b_dictionary_research_3gpu_fast.yaml" \
  "${DICTIONARY_INIT_ADAPTER}"

run_stage \
  "sft_nuosubench_short_3gpu" \
  "configs/sft_qwen3_8b_nuosubench_research_3gpu_fast.yaml"

run_stage \
  "sft_nuosubench_long_3gpu" \
  "configs/sft_qwen3_8b_nuosubench_long_research_3gpu_fast.yaml"

echo "$(timestamp) overnight pipeline completed"
