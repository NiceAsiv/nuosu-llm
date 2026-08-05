#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
base_model="/home/xjtuoss/nuosu-project/models/Qwen3-8B-b968826d"
adapter="$repo_dir/outputs/qwen3-8b-nuosu-balanced-sft-v20260804"

cd "$repo_dir"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec "$repo_dir/.conda/bin/python" -m nuosu_llm.chat \
  --model "$base_model" \
  --adapter "$adapter" \
  --tokenizer "$adapter" \
  --device "${NUOSU_DEVICE:-cuda:2}" \
  --max-input-tokens "${NUOSU_MAX_INPUT_TOKENS:-4096}" \
  --max-new-tokens "${NUOSU_MAX_NEW_TOKENS:-1024}" \
  --temperature "${NUOSU_TEMPERATURE:-0.6}" \
  --top-p "${NUOSU_TOP_P:-0.95}"
