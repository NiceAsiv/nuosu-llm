#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NUM_GPUS="${NUM_GPUS:-1}"
MODEL_DIR="${1:?usage: NUM_GPUS=N run.sh /absolute/verified/model}"
RUN_ID="$(date -u "+%Y%m%dT%H%M%SZ")"
LOG_DIR="${PROJECT_DIR}/artifacts/pipeline/qwen3-1.7b-full/${RUN_ID}"

[[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "NUM_GPUS must be a positive integer" >&2
  exit 2
}
[[ -f "${MODEL_DIR}/VERIFIED.sha256" ]] || {
  echo "verified model marker missing: ${MODEL_DIR}/VERIFIED.sha256" >&2
  exit 2
}
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
  grep -q '[0-9]'; then
  echo "GPU compute processes are active; refusing to overlap the pipeline." >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

run_stage() {
  local stage_name="$1"
  local config_path="$2"
  local output_path="$3"

  if [[ -f "${output_path}/adapter_model.safetensors" ]]; then
    echo "[$(date -u "+%Y-%m-%dT%H:%M:%SZ")] reusing completed ${stage_name}"
    return
  fi
  [[ ! -e "${output_path}" ]] || {
    echo "partial stage output requires inspection: ${output_path}" >&2
    exit 2
  }

  echo "[$(date -u "+%Y-%m-%dT%H:%M:%SZ")] starting ${stage_name}"
  env \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" -m torch.distributed.run \
      --standalone \
      --nproc_per_node="${NUM_GPUS}" \
      scripts/train.py \
      --config "${config_path}" \
      --base-model "${MODEL_DIR}" 2>&1 | tee "${LOG_DIR}/${stage_name}.log"
}

run_stage \
  cpt-corpus \
  recipes/qwen3-1.7b-full/01-cpt-corpus.yaml \
  outputs/qwen3-1.7b-nuosu-cpt
run_stage \
  sft-corpus \
  recipes/qwen3-1.7b-full/02-sft-corpus.yaml \
  outputs/qwen3-1.7b-nuosu-sft

echo "pipeline completed; evaluate with a separately maintained blind evaluation set"
