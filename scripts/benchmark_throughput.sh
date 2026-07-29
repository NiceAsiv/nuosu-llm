#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/xjtuoss/nuosu-project/nuosu-llm}"
PYTHON_BIN="${PYTHON_BIN:-/home/xjtuoss/nuosu-project/.venv/bin/python}"
CONFIG_PATH="${1:?usage: BASE_MODEL=/verified/model benchmark_throughput.sh CONFIG [STEPS] [INIT_ADAPTER]}"
MAX_STEPS="${2:-30}"
INIT_ADAPTER="${3:-}"
BASE_MODEL="${BASE_MODEL:?set BASE_MODEL to a local directory containing VERIFIED.sha256}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${PROJECT_DIR}/outputs/benchmarks/${RUN_ID}"
LOG_DIR="${PROJECT_DIR}/logs/benchmarks"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
cd "${PROJECT_DIR}"

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU compute processes are active; stop or checkpoint them before benchmarking." >&2
  exit 2
fi

command=(
  env
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
  HF_DATASETS_OFFLINE=1
  CUDA_VISIBLE_DEVICES=0,1,2
  OMP_NUM_THREADS=1
  PYTHONUNBUFFERED=1
  "${PYTHON_BIN}" -m torch.distributed.run
  --standalone
  --nproc_per_node=3
  scripts/train.py
  --config "${CONFIG_PATH}"
  --base-model "${BASE_MODEL}"
  --output-dir "${OUTPUT_DIR}"
  --max-steps "${MAX_STEPS}"
)

if [[ -n "${INIT_ADAPTER}" ]]; then
  command+=(--init-adapter "${INIT_ADAPTER}")
fi

echo "benchmark=${RUN_ID} config=${CONFIG_PATH} steps=${MAX_STEPS}"
"${command[@]}" 2>&1 | tee "${LOG_PATH}"
echo "benchmark_log=${LOG_PATH}"
