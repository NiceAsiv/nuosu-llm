#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NUOSU_BASE_MODEL="${NUOSU_BASE_MODEL:-/home/xjtuoss/nuosu-project/models/Qwen3-1.7B-Base-36be17a0}"
export NUOSU_ADAPTER="${NUOSU_ADAPTER:-${repo_dir}/outputs/qwen3-1.7b-nuosu-mt-sft-v20260805}"
export NUOSU_TOKENIZER="${NUOSU_TOKENIZER:-${NUOSU_ADAPTER}}"

exec "${PYTHON_BIN:-${repo_dir}/.conda/bin/python}" -m nuosu_llm.translate "$@"
