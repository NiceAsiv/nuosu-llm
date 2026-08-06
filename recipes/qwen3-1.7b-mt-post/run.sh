#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.conda/bin/python}"
NUM_GPUS="${NUM_GPUS:-3}"
MODEL_DIR="${1:-${NUOSU_BASE_MODEL:-/home/xjtuoss/nuosu-project/models/Qwen3-1.7B-70d244cc}}"
CORPUS_DIR="${NUOSU_CORPUS_DIR:-/home/xjtuoss/nuosu-project/nuosu-corpus/data/task_corpus_20260804_v2}"
DATASET_DIR="${PROJECT_DIR}/artifacts/datasets/nuosu-mt-v20260804"
SFT_OUTPUT="${NUOSU_SFT_OUTPUT:-${PROJECT_DIR}/outputs/qwen3-1.7b-nuosu-mt-post-sft-v20260806}"
RUN_ID="${NUOSU_RUN_ID:-$(date -u '+%Y%m%dT%H%M%SZ')}"
RUN_DIR="${PROJECT_DIR}/artifacts/pipeline/qwen3-1.7b-mt-post/${RUN_ID}"
EVAL_DIR="${RUN_DIR}/evaluation"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
EVAL_BASELINE="${EVAL_BASELINE:-1}"

export CUDA_VISIBLE_DEVICES HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${RUN_DIR}" "${EVAL_DIR}"
exec > >(tee -a "${RUN_DIR}/pipeline.log") 2>&1
on_error() {
  local exit_code=$?
  printf '%s\n' "failed_at_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
    "exit_code=${exit_code}" > "${RUN_DIR}/FAILED"
  exit "${exit_code}"
}
trap on_error ERR

[[ -f "${MODEL_DIR}/VERIFIED.sha256" ]] || {
  echo "verified post-trained model missing: ${MODEL_DIR}" >&2
  exit 2
}
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU compute processes are active; refusing to overlap the pipeline." >&2
  exit 2
fi

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" scripts/prepare_mt_dataset.py \
  --train "${CORPUS_DIR}/ready_sft.jsonl" \
  --validation "${CORPUS_DIR}/validation_sft.jsonl" \
  --test "${CORPUS_DIR}/research_test_eval.jsonl" \
  --output-dir "${DATASET_DIR}" > "${RUN_DIR}/prepare-mt-dataset.json"

"${PYTHON_BIN}" scripts/capture_run_metadata.py \
  --experiment-id "qwen3-1.7b-mt-post-${RUN_ID}" \
  --output "${RUN_DIR}/run-manifest.json" \
  --input "${MODEL_DIR}/snapshot_manifest.json" \
  --input "${CORPUS_DIR}/manifest.json" \
  --input "${DATASET_DIR}/manifest.json" \
  --input "${SCRIPT_DIR}/sft.yaml" \
  --launch-command "NUM_GPUS=${NUM_GPUS} bash recipes/qwen3-1.7b-mt-post/run.sh ${MODEL_DIR}"

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] running recipe-matched MT gate"
bash "${SCRIPT_DIR}/gate.sh" "${MODEL_DIR}"

latest_checkpoint() {
  find "$1" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' -printf '%p\n' 2>/dev/null |
    sort -V | tail -n 1
}

declare -a train_args=(
  scripts/train.py
  --config "${SCRIPT_DIR}/sft.yaml"
  --base-model "${MODEL_DIR}"
  --train-file "${DATASET_DIR}/train.jsonl"
  --eval-file "${DATASET_DIR}/validation.jsonl"
  --output-dir "${SFT_OUTPUT}"
)
if [[ ! -f "${SFT_OUTPUT}/adapter_model.safetensors" ]]; then
  if [[ -d "${SFT_OUTPUT}" ]]; then
    checkpoint="$(latest_checkpoint "${SFT_OUTPUT}")"
    [[ -n "${checkpoint}" ]] || {
      echo "partial SFT output has no resumable checkpoint: ${SFT_OUTPUT}" >&2
      exit 2
    }
    train_args+=(--resume-from-checkpoint "${checkpoint}")
  fi
  "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node="${NUM_GPUS}" \
    "${train_args[@]}" 2>&1 | tee "${RUN_DIR}/sft.log"
fi

run_eval() {
  local label="$1"
  local input="$2"
  local output="$3"
  local max_new_tokens="$4"
  shift 4
  if [[ ! -f "${output}" ]]; then
    "${PYTHON_BIN}" -m torch.distributed.run \
      --standalone --nproc_per_node="${NUM_GPUS}" \
      scripts/evaluate_prompts.py \
      --model "${MODEL_DIR}" --input "${input}" --output "${output}" \
      --batch-size 8 --max-input-tokens 1024 --max-new-tokens "${max_new_tokens}" \
      --thinking-mode no_think --resume "$@" 2>&1 | tee "${RUN_DIR}/evaluate-${label}.log"
  fi
  "${PYTHON_BIN}" scripts/score_evaluation.py \
    --input "${output}" --label "${label}" \
    --output-json "${EVAL_DIR}/${label}-metrics.json" \
    --output-markdown "${EVAL_DIR}/${label}-metrics.md"
}

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] running 512-record collapse canary"
CANARY_INPUT="${RUN_DIR}/canary.jsonl"
head -n 512 "${DATASET_DIR}/test.jsonl" > "${CANARY_INPUT}"
run_eval post-canary "${CANARY_INPUT}" "${EVAL_DIR}/post-canary-generations.jsonl" 128 \
  --adapter "${SFT_OUTPUT}" --tokenizer "${SFT_OUTPUT}"
"${PYTHON_BIN}" scripts/gates/check_overfit_metrics.py \
  --metrics "${EVAL_DIR}/post-canary-metrics.json" \
  --min-compact-exact 0 --min-chrf2 0 \
  --max-replacement-rate 0.01 --max-length-stop-rate 0.05 \
  --result-output "${RUN_DIR}/canary-check.json"

if [[ "${EVAL_BASELINE}" == "1" ]]; then
  run_eval base "${DATASET_DIR}/test.jsonl" "${EVAL_DIR}/base-generations.jsonl" 512
fi
run_eval post-mt "${DATASET_DIR}/test.jsonl" "${EVAL_DIR}/post-mt-generations.jsonl" 512 \
  --adapter "${SFT_OUTPUT}" --tokenizer "${SFT_OUTPUT}"

"${PYTHON_BIN}" scripts/inspect_adapter.py "${SFT_OUTPUT}" \
  --output "${RUN_DIR}/adapter-inspection.json"
sha256sum "${SFT_OUTPUT}/adapter_model.safetensors" \
  "${SFT_OUTPUT}/tokenizer_expansion.json" \
  "${DATASET_DIR}/manifest.json" \
  "${EVAL_DIR}/post-mt-generations.jsonl" \
  "${EVAL_DIR}/post-mt-metrics.json" > "${RUN_DIR}/SHA256SUMS"
printf '%s\n' "completed_at_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
  "adapter=${SFT_OUTPUT}" \
  "evaluation=${EVAL_DIR}/post-mt-metrics.json" > "${RUN_DIR}/COMPLETED"
echo "pipeline completed: ${RUN_DIR}"
