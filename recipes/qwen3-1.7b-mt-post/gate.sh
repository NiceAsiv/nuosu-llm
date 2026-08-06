#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.conda/bin/python}"
MODEL_DIR="${1:?usage: gate.sh /absolute/verified/Qwen3-1.7B}"
SOURCE_DATA="${SOURCE_DATA:-${PROJECT_DIR}/artifacts/datasets/nuosu-mt-v20260804/train.jsonl}"
GATE_DIR="${GATE_DIR:-${PROJECT_DIR}/artifacts/gates/qwen3-1.7b-mt-post-256}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/gates/qwen3-1.7b-mt-post-256}"
MAX_STEPS="${MAX_STEPS:-400}"

export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p "${GATE_DIR}" "$(dirname -- "${OUTPUT_DIR}")"

on_error() {
  local exit_code=$?
  printf '%s\n' "failed_at_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
    "exit_code=${exit_code}" > "${GATE_DIR}/FAILED"
  exit "${exit_code}"
}
trap on_error ERR

[[ -f "${MODEL_DIR}/VERIFIED.sha256" ]] || {
  echo "verified model marker missing: ${MODEL_DIR}/VERIFIED.sha256" >&2
  exit 2
}
[[ -f "${SOURCE_DATA}" ]] || {
  echo "MT source data missing: ${SOURCE_DATA}" >&2
  exit 2
}
if [[ -f "${GATE_DIR}/PASSED" && -f "${OUTPUT_DIR}/adapter_model.safetensors" ]]; then
  echo "reusing passed gate: ${GATE_DIR}"
  exit 0
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "gate output exists without PASSED marker: ${OUTPUT_DIR}" >&2
  exit 2
fi

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" scripts/gates/prepare_sft_overfit.py \
  --input "${SOURCE_DATA}" \
  --train-output "${GATE_DIR}/train.jsonl" \
  --eval-output "${GATE_DIR}/eval.jsonl" \
  --limit 256 \
  --balanced-directions 'zh->ii' 'ii->zh' 'en->ii' 'ii->en'

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone --nproc_per_node=3 \
  scripts/train.py \
  --config "${SCRIPT_DIR}/sft.yaml" \
  --base-model "${MODEL_DIR}" \
  --train-file "${GATE_DIR}/train.jsonl" \
  --eval-file "${GATE_DIR}/train.jsonl" \
  --output-dir "${OUTPUT_DIR}" \
  --max-steps "${MAX_STEPS}" 2>&1 | tee "${GATE_DIR}/training.log"

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" scripts/evaluate_prompts.py \
  --model "${MODEL_DIR}" \
  --adapter "${OUTPUT_DIR}" \
  --tokenizer "${OUTPUT_DIR}" \
  --input "${GATE_DIR}/eval.jsonl" \
  --output "${GATE_DIR}/generations.jsonl" \
  --batch-size 8 \
  --max-input-tokens 512 \
  --max-new-tokens 512 \
  --thinking-mode no_think

"${PYTHON_BIN}" scripts/score_evaluation.py \
  --input "${GATE_DIR}/generations.jsonl" \
  --label qwen3-1.7b-mt-post-256 \
  --output-json "${GATE_DIR}/metrics.json" \
  --output-markdown "${GATE_DIR}/metrics.md"
"${PYTHON_BIN}" scripts/gates/check_overfit_metrics.py \
  --metrics "${GATE_DIR}/metrics.json" \
  --min-compact-exact 0.90 \
  --min-chrf2 90 \
  --max-replacement-rate 0.01 \
  --max-length-stop-rate 0.05 \
  --result-output "${GATE_DIR}/gate-check.json"

printf '%s\n' "${MODEL_DIR}" > "${GATE_DIR}/PASSED"
echo "MT post-trained gate passed: ${GATE_DIR}"
