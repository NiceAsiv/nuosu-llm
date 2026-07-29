#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_DIR="${1:?usage: run_research_pipeline.sh /absolute/verified/model}"
RUN_ID="$(date -u "+%Y%m%dT%H%M%SZ")"
LOG_DIR="${PROJECT_DIR}/artifacts/pipeline/${RUN_ID}"
GATE_MARKER="${PROJECT_DIR}/artifacts/gates/sft-overfit-64/PASSED"

[[ -f "${MODEL_DIR}/VERIFIED.sha256" ]] || {
  echo "verified model marker missing: ${MODEL_DIR}/VERIFIED.sha256" >&2
  exit 2
}
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU compute processes are active; refusing to overlap the pipeline." >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

run_stage() {
  local stage_name="$1"
  local config_path="$2"
  local output_path="$3"

  [[ ! -e "${output_path}" ]] || {
    echo "stage output already exists: ${output_path}" >&2
    exit 2
  }
  echo "[$(date -u "+%Y-%m-%dT%H:%M:%SZ")] starting ${stage_name}"
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
      --config "${config_path}" \
      --base-model "${MODEL_DIR}" 2>&1 | tee "${LOG_DIR}/${stage_name}.log"
}

if [[ -f "${GATE_MARKER}" ]] && [[ "$(cat -- "${GATE_MARKER}")" == "${MODEL_DIR}" ]]; then
  echo "using passed SFT overfit gate: ${GATE_MARKER}"
else
  bash scripts/gates/run_sft_overfit_64.sh "${MODEL_DIR}" \
    2>&1 | tee "${LOG_DIR}/gate-sft-overfit-64.log"
fi

run_stage \
  cpt-ocr-gt \
  configs/cpt_qwen3_8b_ocr_gt_research_3gpu.yaml \
  outputs/qwen3-8b-nuosu-ocr-gt-cpt-3gpu-stable
run_stage \
  sft-dictionary \
  configs/sft_qwen3_8b_dictionary_research_3gpu_fast.yaml \
  outputs/qwen3-8b-nuosu-dictionary-sft-3gpu-fast
run_stage \
  sft-nuosubench-short \
  configs/sft_qwen3_8b_nuosubench_research_3gpu_fast.yaml \
  outputs/qwen3-8b-nuosu-nuosubench-short-sft-3gpu-fast
run_stage \
  sft-nuosubench-long \
  configs/sft_qwen3_8b_nuosubench_long_research_3gpu_fast.yaml \
  outputs/qwen3-8b-nuosu-nuosubench-sft-3gpu-fast

echo "pipeline completed; run validation generation before any held-out test"
