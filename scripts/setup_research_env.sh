#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
ENV_PREFIX="${1:-${PROJECT_DIR}/.conda}"
CONDA_BIN="${CONDA_BIN:-${HOME}/miniconda3/bin/conda}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
ARTIFACT_DIR="${PROJECT_DIR}/artifacts/environment"

[[ -x "${CONDA_BIN}" ]] || {
  echo "conda executable not found: ${CONDA_BIN}" >&2
  exit 2
}

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  "${CONDA_BIN}" create --yes --prefix "${ENV_PREFIX}" \
    "python=${PYTHON_VERSION}" pip
fi

PYTHON_BIN="${ENV_PREFIX}/bin/python"
"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
"${PYTHON_BIN}" -m pip install \
  "torch==${TORCH_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"
"${PYTHON_BIN}" -m pip install -e "${PROJECT_DIR}[train,dev]"
"${PYTHON_BIN}" -m pip check

mkdir -p "${ARTIFACT_DIR}"
"${PYTHON_BIN}" -m pip freeze >"${ARTIFACT_DIR}/pip-freeze.txt"
nvidia-smi -q >"${ARTIFACT_DIR}/nvidia-smi-q.txt"
git -C "${PROJECT_DIR}" rev-parse HEAD >"${ARTIFACT_DIR}/code-revision.txt"
"${PYTHON_BIN}" - <<'PY' >"${ARTIFACT_DIR}/runtime-versions.txt"
import platform

import accelerate
import bitsandbytes
import datasets
import peft
import torch
import transformers
import trl

print("python", platform.python_version())
print("torch", torch.__version__)
print("cuda_runtime", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_devices", torch.cuda.device_count())
print("transformers", transformers.__version__)
print("trl", trl.__version__)
print("peft", peft.__version__)
print("datasets", datasets.__version__)
print("accelerate", accelerate.__version__)
print("bitsandbytes", bitsandbytes.__version__)
PY

cat "${ARTIFACT_DIR}/runtime-versions.txt"
echo "research environment ready: ${ENV_PREFIX}"
