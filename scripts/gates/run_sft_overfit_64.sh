#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_DIR="${1:?usage: run_sft_overfit_64.sh /absolute/verified/model}"
SOURCE_DATA="${SOURCE_DATA:-${PROJECT_DIR}/../nuosu-corpus/data/processed/yixueyanjiu_dictionary_sft/train.jsonl}"
GATE_DIR="${GATE_DIR:-${PROJECT_DIR}/artifacts/gates/sft-overfit-64}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/gates/qwen3-8b-sft-overfit-64}"
CONFIG_PATH="${PROJECT_DIR}/configs/gates/sft_overfit_64.yaml"
TRAIN_FILE="${GATE_DIR}/train.jsonl"
EVAL_FILE="${GATE_DIR}/eval.jsonl"
GEN_FILE="${GATE_DIR}/generations.jsonl"
METRICS_FILE="${GATE_DIR}/metrics.json"
LOG_FILE="${GATE_DIR}/training.log"

[[ -f "${MODEL_DIR}/VERIFIED.sha256" ]] || {
  echo "verified model marker missing: ${MODEL_DIR}/VERIFIED.sha256" >&2
  exit 2
}
[[ -f "${SOURCE_DATA}" ]] || {
  echo "gate source data missing: ${SOURCE_DATA}" >&2
  exit 2
}
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU compute processes are active; refusing to overlap the gate." >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "gate output already exists; move it before rerunning: ${OUTPUT_DIR}" >&2
  exit 2
fi

mkdir -p "${GATE_DIR}" "$(dirname -- "${OUTPUT_DIR}")"
cd "${PROJECT_DIR}"

"${PYTHON_BIN}" scripts/gates/prepare_sft_overfit.py \
  --input "${SOURCE_DATA}" \
  --train-output "${TRAIN_FILE}" \
  --eval-output "${EVAL_FILE}" \
  --limit 64

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
    --config "${CONFIG_PATH}" \
    --base-model "${MODEL_DIR}" \
    --train-file "${TRAIN_FILE}" \
    --eval-file "${TRAIN_FILE}" \
    --output-dir "${OUTPUT_DIR}" 2>&1 | tee "${LOG_FILE}"

env \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0,1,2 \
  OMP_NUM_THREADS=1 \
  "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=3 \
    scripts/evaluate_prompts.py \
    --model "${MODEL_DIR}" \
    --adapter "${OUTPUT_DIR}" \
    --tokenizer "${OUTPUT_DIR}" \
    --input "${EVAL_FILE}" \
    --output "${GEN_FILE}" \
    --batch-size 8 \
    --max-input-tokens 128 \
    --max-new-tokens 64

"${PYTHON_BIN}" scripts/score_evaluation.py \
  --input "${GEN_FILE}" \
  --label qwen3-8b-sft-overfit-64 \
  --output-json "${METRICS_FILE}" \
  --output-markdown "${GATE_DIR}/metrics.md"

"${PYTHON_BIN}" scripts/gates/check_overfit_metrics.py \
  --metrics "${METRICS_FILE}"

printf '%s\n' "${MODEL_DIR}" > "${GATE_DIR}/PASSED"
echo "SFT overfit gate passed: ${GATE_DIR}"
