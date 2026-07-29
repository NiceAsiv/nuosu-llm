#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
RECOVERY_PID="${1:?usage: wait_for_recovery_then_train.sh PID MODEL_DIR RECOVERY_LOG}"
MODEL_DIR="${2:?usage: wait_for_recovery_then_train.sh PID MODEL_DIR RECOVERY_LOG}"
RECOVERY_LOG="${3:?usage: wait_for_recovery_then_train.sh PID MODEL_DIR RECOVERY_LOG}"

echo "waiting for recovery PID ${RECOVERY_PID}"
while kill -0 "${RECOVERY_PID}" 2>/dev/null; do
  sleep 30
done

if ! grep -Fq "recovery passed" "${RECOVERY_LOG}"; then
  echo "recovery did not pass; training will not start: ${RECOVERY_LOG}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_DIR}/VERIFIED.sha256" ]]; then
  echo "verified marker is missing; training will not start: ${MODEL_DIR}" >&2
  exit 1
fi

echo "recovery passed; starting gated research pipeline"
exec bash "${PROJECT_DIR}/experiments/three-gpu-24gb/run_pipeline.sh" "${MODEL_DIR}"
