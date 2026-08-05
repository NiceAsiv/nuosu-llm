#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.conda/bin/python}"
NUM_GPUS="${NUM_GPUS:-3}"
MODEL_DIR="${1:-${NUOSU_BASE_MODEL:-/home/xjtuoss/nuosu-project/models/Qwen3-1.7B-Base-36be17a0}}"
CORPUS_DIR="${NUOSU_CORPUS_DIR:-/home/xjtuoss/nuosu-project/nuosu-corpus/data/task_corpus_20260804_v2}"
DATASET_DIR="${PROJECT_DIR}/artifacts/datasets/nuosu-mt-v20260804"
CPT_OUTPUT="${NUOSU_CPT_OUTPUT:-${PROJECT_DIR}/outputs/qwen3-1.7b-nuosu-mt-cpt-v20260805}"
SFT_OUTPUT="${NUOSU_SFT_OUTPUT:-${PROJECT_DIR}/outputs/qwen3-1.7b-nuosu-mt-sft-v20260805}"
RUN_ID="${NUOSU_RUN_ID:-$(date -u '+%Y%m%dT%H%M%SZ')}"
RUN_DIR="${PROJECT_DIR}/artifacts/pipeline/qwen3-1.7b-mt/${RUN_ID}"
EVAL_DIR="${RUN_DIR}/evaluation"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
EVAL_BASELINE="${EVAL_BASELINE:-1}"

export CUDA_VISIBLE_DEVICES
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${RUN_DIR}" "${EVAL_DIR}"
exec > >(tee -a "${RUN_DIR}/pipeline.log") 2>&1

on_error() {
  local exit_code=$?
  printf '%s\n' "failed_at_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "exit_code=${exit_code}" > "${RUN_DIR}/FAILED"
  echo "pipeline failed; inspect ${RUN_DIR}/pipeline.log" >&2
  exit "${exit_code}"
}
trap on_error ERR

require_file() {
  [[ -f "$1" ]] || {
    echo "required file missing: $1" >&2
    exit 2
  }
}

[[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "NUM_GPUS must be a positive integer" >&2
  exit 2
}
require_file "${MODEL_DIR}/VERIFIED.sha256"
require_file "${CORPUS_DIR}/ready_cpt.jsonl"
require_file "${CORPUS_DIR}/ready_sft.jsonl"
require_file "${CORPUS_DIR}/validation_sft.jsonl"
require_file "${CORPUS_DIR}/research_test_eval.jsonl"
require_file "${CORPUS_DIR}/manifest.json"

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU compute processes are active; refusing to overlap the pipeline." >&2
  exit 2
fi

cd "${PROJECT_DIR}"
echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] preparing target-only MT data"
"${PYTHON_BIN}" scripts/prepare_mt_dataset.py \
  --train "${CORPUS_DIR}/ready_sft.jsonl" \
  --validation "${CORPUS_DIR}/validation_sft.jsonl" \
  --test "${CORPUS_DIR}/research_test_eval.jsonl" \
  --output-dir "${DATASET_DIR}" \
  > "${RUN_DIR}/prepare-mt-dataset.json"

"${PYTHON_BIN}" scripts/capture_run_metadata.py \
  --experiment-id "qwen3-1.7b-mt-${RUN_ID}" \
  --output "${RUN_DIR}/run-manifest.json" \
  --input "${CORPUS_DIR}/manifest.json" \
  --input "${DATASET_DIR}/manifest.json" \
  --input "${SCRIPT_DIR}/01-cpt.yaml" \
  --input "${SCRIPT_DIR}/02-sft.yaml" \
  --launch-command "NUM_GPUS=${NUM_GPUS} bash recipes/qwen3-1.7b-mt/run.sh ${MODEL_DIR}"

latest_checkpoint() {
  find "$1" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' -printf '%p\n' 2>/dev/null |
    sort -V |
    tail -n 1
}

run_training_stage() {
  local stage_name="$1"
  local config_path="$2"
  local output_path="$3"
  local train_path="$4"
  local eval_path="${5:-}"
  local init_adapter="${6:-}"
  local -a args=(
    scripts/train.py
    --config "${config_path}"
    --base-model "${MODEL_DIR}"
    --train-file "${train_path}"
    --output-dir "${output_path}"
  )

  if [[ -f "${output_path}/adapter_model.safetensors" ]]; then
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] reusing completed ${stage_name}: ${output_path}"
    return
  fi
  if [[ -n "${eval_path}" ]]; then
    args+=(--eval-file "${eval_path}")
  fi
  if [[ -n "${init_adapter}" ]]; then
    args+=(--init-adapter "${init_adapter}")
  fi
  if [[ -d "${output_path}" ]]; then
    local checkpoint
    checkpoint="$(latest_checkpoint "${output_path}")"
    [[ -n "${checkpoint}" ]] || {
      echo "partial output has no resumable checkpoint: ${output_path}" >&2
      exit 2
    }
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] resuming ${stage_name}: ${checkpoint}"
    args+=(--resume-from-checkpoint "${checkpoint}")
  else
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] starting ${stage_name}"
  fi

  "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    "${args[@]}" 2>&1 | tee "${RUN_DIR}/${stage_name}.log"
  require_file "${output_path}/adapter_model.safetensors"
}

run_training_stage \
  cpt \
  "${SCRIPT_DIR}/01-cpt.yaml" \
  "${CPT_OUTPUT}" \
  "${CORPUS_DIR}/ready_cpt.jsonl"

run_training_stage \
  sft \
  "${SCRIPT_DIR}/02-sft.yaml" \
  "${SFT_OUTPUT}" \
  "${DATASET_DIR}/train.jsonl" \
  "${DATASET_DIR}/validation.jsonl" \
  "${CPT_OUTPUT}"

run_evaluation() {
  local label="$1"
  local output_path="${EVAL_DIR}/${label}-generations.jsonl"
  shift
  if [[ ! -f "${output_path}" ]]; then
    "${PYTHON_BIN}" -m torch.distributed.run \
      --standalone \
      --nproc_per_node="${NUM_GPUS}" \
      scripts/evaluate_prompts.py \
      --model "${MODEL_DIR}" \
      --input "${DATASET_DIR}/test.jsonl" \
      --output "${output_path}" \
      --batch-size 8 \
      --max-input-tokens 1024 \
      --max-new-tokens 512 \
      --thinking-mode no_think \
      --resume \
      "$@" 2>&1 | tee "${RUN_DIR}/evaluate-${label}.log"
  fi
  "${PYTHON_BIN}" scripts/score_evaluation.py \
    --input "${output_path}" \
    --label "${label}" \
    --output-json "${EVAL_DIR}/${label}-metrics.json" \
    --output-markdown "${EVAL_DIR}/${label}-metrics.md" \
    > "${RUN_DIR}/score-${label}.log"
}

if [[ "${EVAL_BASELINE}" == "1" ]]; then
  run_evaluation base
fi
run_evaluation expanded-mt --adapter "${SFT_OUTPUT}" --tokenizer "${SFT_OUTPUT}"

"${PYTHON_BIN}" scripts/inspect_adapter.py "${CPT_OUTPUT}" "${SFT_OUTPUT}" \
  --output "${RUN_DIR}/adapter-inspection.json"
sha256sum \
  "${SFT_OUTPUT}/adapter_model.safetensors" \
  "${SFT_OUTPUT}/tokenizer_expansion.json" \
  "${DATASET_DIR}/manifest.json" \
  "${EVAL_DIR}/expanded-mt-generations.jsonl" \
  "${EVAL_DIR}/expanded-mt-metrics.json" \
  > "${RUN_DIR}/SHA256SUMS"

printf '%s\n' \
  "completed_at_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
  "adapter=${SFT_OUTPUT}" \
  "evaluation=${EVAL_DIR}/expanded-mt-metrics.json" \
  > "${RUN_DIR}/COMPLETED"
echo "pipeline completed: ${RUN_DIR}"
