# Qwen3-1.7B full Nuosu recipe

This recipe trains a smaller BF16 LoRA model on the pinned task-oriented corpus:

1. continued pretraining on all 4,238 usable `ready_cpt.jsonl` records;
2. supervised fine-tuning on all 113,269 usable `ready_sft.jsonl` records.

Both files come from `NiceAsiv/nuosu-corpus` revision `v2026.08.02`. Run
`python scripts/download_training_corpus.py` before starting. The corpus snapshot
does not contain a held-out validation split, so evaluation must use a separately
maintained blind set.

Run on one or more GPUs with the same entry point:

```bash
NUM_GPUS=1 bash recipes/qwen3-1.7b-full/run.sh /absolute/path/to/verified/model

CUDA_VISIBLE_DEVICES=0,1,2 NUM_GPUS=3 \
  bash recipes/qwen3-1.7b-full/run.sh /absolute/path/to/verified/model
```

The runner refuses to overlap an existing GPU job. A completed stage is reused
only when its top-level `adapter_model.safetensors` exists; a partial output
directory stops the pipeline for manual inspection.
