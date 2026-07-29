#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_ID="${REPO_ID:-Qwen/Qwen3-8B-Base}"
REVISION="${REVISION:-49e3418fbbbca6ecbdf9608b4d22e5a407081db4}"
PROXY_URL="${PROXY_URL:-http://127.0.0.1:7897}"
MODEL_ROOT="${MODEL_ROOT:-$(cd -- "${PROJECT_DIR}/.." && pwd)/models}"
MODEL_DIR="${MODEL_DIR:-${MODEL_ROOT}/Qwen3-8B-Base-${REVISION:0:12}}"
QUARANTINE_ROOT="${QUARANTINE_ROOT:-$(cd -- "${PROJECT_DIR}/.." && pwd)/quarantine}"
SOURCE_CACHE="${SOURCE_CACHE:-${HOME}/.cache/huggingface/hub/models--Qwen--Qwen3-8B-Base}"
CHECKSUM_FILE="${CHECKSUM_FILE:-${SCRIPT_DIR}/manifests/qwen3-8b-base-49e3418.sha256}"
EXECUTE=0
QUARANTINE_OUTPUTS=1
FORCE_DOWNLOAD=1
RUN_SMOKE=1

usage() {
  cat <<'EOF'
Usage:
  bash scripts/model/recover_qwen3_base.sh              # dry-run
  bash scripts/model/recover_qwen3_base.sh --execute    # quarantine, download, verify, smoke-test

Options:
  --execute             Perform changes. Without this flag the script is read-only.
  --keep-outputs        Do not quarantine outputs/qwen3-8b-* adapters.
  --no-force-download   Allow Hugging Face to reuse files in the isolated download cache.
  --skip-smoke          Stop after SHA-256 verification.
  --help                Show this help.

Environment overrides:
  PROJECT_DIR, PYTHON_BIN, PROXY_URL, MODEL_ROOT, MODEL_DIR,
  QUARANTINE_ROOT, SOURCE_CACHE, REPO_ID, REVISION, CHECKSUM_FILE.
EOF
}

while (($#)); do
  case "$1" in
    --execute) EXECUTE=1 ;;
    --keep-outputs) QUARANTINE_OUTPUTS=0 ;;
    --no-force-download) FORCE_DOWNLOAD=0 ;;
    --skip-smoke) RUN_SMOKE=0 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

timestamp() {
  date -u "+%Y-%m-%dT%H:%M:%SZ"
}

log() {
  echo "[$(timestamp)] $*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "required file not found: $1"
}

require_file "${CHECKSUM_FILE}"
require_file "${PROJECT_DIR}/scripts/model/download_snapshot.py"
require_file "${PROJECT_DIR}/scripts/model/smoke_test_base.py"
command -v sha256sum >/dev/null || die "sha256sum is required"
[[ -x "${PYTHON_BIN}" ]] || command -v "${PYTHON_BIN}" >/dev/null || \
  die "Python is not executable: ${PYTHON_BIN}"

case "${SOURCE_CACHE}" in
  "${HOME}"/.cache/huggingface/*) ;;
  *) die "SOURCE_CACHE must stay under ${HOME}/.cache/huggingface: ${SOURCE_CACHE}" ;;
esac
case "${MODEL_ROOT}" in
  "${HOME}"/*) ;;
  *) die "MODEL_ROOT must stay under ${HOME}: ${MODEL_ROOT}" ;;
esac
case "${QUARANTINE_ROOT}" in
  "${HOME}"/*) ;;
  *) die "QUARANTINE_ROOT must stay under ${HOME}: ${QUARANTINE_ROOT}" ;;
esac

RUN_ID="$(date -u "+%Y%m%dT%H%M%SZ")"
QUARANTINE_DIR="${QUARANTINE_ROOT}/${RUN_ID}"
REPORT_DIR="${PROJECT_DIR}/artifacts/model-recovery/${RUN_ID}"

log "mode=$([[ ${EXECUTE} -eq 1 ]] && echo execute || echo dry-run)"
log "repo=${REPO_ID} revision=${REVISION}"
log "corrupt_cache=${SOURCE_CACHE}"
log "model_dir=${MODEL_DIR}"
log "quarantine_dir=${QUARANTINE_DIR}"
log "proxy=${PROXY_URL}"

OUTPUT_TARGETS=()
if [[ ${QUARANTINE_OUTPUTS} -eq 1 ]]; then
  shopt -s nullglob
  OUTPUT_TARGETS=("${PROJECT_DIR}"/outputs/qwen3-8b-*)
  shopt -u nullglob
fi

if [[ ${EXECUTE} -eq 0 ]]; then
  [[ -e "${SOURCE_CACHE}" ]] && log "would move cache: ${SOURCE_CACHE}" || \
    log "cache already absent: ${SOURCE_CACHE}"
  for target in "${OUTPUT_TARGETS[@]}"; do
    log "would move derived output: ${target}"
  done
  log "would download into: ${MODEL_DIR}"
  log "would verify with: ${CHECKSUM_FILE}"
  [[ ${RUN_SMOKE} -eq 1 ]] && log "would run three-prompt generation smoke test"
  log "dry-run complete; re-run with --execute"
  exit 0
fi

mkdir -p \
  "${QUARANTINE_DIR}/cache" \
  "${QUARANTINE_DIR}/outputs" \
  "${QUARANTINE_DIR}/models" \
  "${REPORT_DIR}" \
  "${MODEL_ROOT}"

if [[ -e "${SOURCE_CACHE}" ]]; then
  log "quarantining corrupt Hugging Face cache"
  mv -- "${SOURCE_CACHE}" "${QUARANTINE_DIR}/cache/"
fi

for target in "${OUTPUT_TARGETS[@]}"; do
  [[ -e "${target}" ]] || continue
  log "quarantining derived output $(basename -- "${target}")"
  mv -- "${target}" "${QUARANTINE_DIR}/outputs/"
done

if [[ -e "${MODEL_DIR}" ]]; then
  log "quarantining pre-existing destination model"
  mv -- "${MODEL_DIR}" "${QUARANTINE_DIR}/models/"
fi

DOWNLOAD_COMMAND=(
  "${PYTHON_BIN}"
  "${PROJECT_DIR}/scripts/model/download_snapshot.py"
  --repo-id "${REPO_ID}"
  --revision "${REVISION}"
  --output-dir "${MODEL_DIR}"
)
if [[ ${FORCE_DOWNLOAD} -eq 1 ]]; then
  DOWNLOAD_COMMAND+=(--force-download)
fi

log "downloading pinned snapshot through the configured proxy"
env \
  HTTP_PROXY="${PROXY_URL}" \
  HTTPS_PROXY="${PROXY_URL}" \
  http_proxy="${PROXY_URL}" \
  https_proxy="${PROXY_URL}" \
  HF_HOME="${MODEL_ROOT}/.hf-clean-cache" \
  HF_HUB_DISABLE_XET=1 \
  HF_HUB_DOWNLOAD_TIMEOUT=300 \
  "${DOWNLOAD_COMMAND[@]}" | tee "${REPORT_DIR}/download.jsonl"

log "verifying actual model bytes"
(
  cd "${MODEL_DIR}"
  sha256sum --check "${CHECKSUM_FILE}"
) | tee "${REPORT_DIR}/sha256.txt"
cp -- "${CHECKSUM_FILE}" "${MODEL_DIR}/VERIFIED.sha256"

if [[ ${RUN_SMOKE} -eq 1 ]]; then
  log "running untouched-model generation gate on cuda:0"
  CUDA_VISIBLE_DEVICES=0 \
    "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/model/smoke_test_base.py" \
      --model "${MODEL_DIR}" \
      --output "${REPORT_DIR}/smoke-test.json"
fi

cat > "${REPORT_DIR}/recovery.env" <<EOF
REPO_ID=${REPO_ID}
REVISION=${REVISION}
MODEL_DIR=${MODEL_DIR}
QUARANTINE_DIR=${QUARANTINE_DIR}
CHECKSUM_FILE=${CHECKSUM_FILE}
COMPLETED_AT=$(timestamp)
EOF

log "recovery passed"
log "verified_model=${MODEL_DIR}"
log "quarantine=${QUARANTINE_DIR}"
log "report=${REPORT_DIR}"
log "training was not started; run the 64-example overfit gate next"
