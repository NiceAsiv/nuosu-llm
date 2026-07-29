# Qwen3-1.7B full Nuosu recipe

This recipe mirrors the validated 8B data route with a smaller BF16 LoRA model:

1. OCR ground-truth continued pretraining;
2. dictionary SFT;
3. NuosuBench short-sequence SFT;
4. NuosuBench long-tail SFT.

The fourth stage contains longer benchmark-derived examples, not a dedicated
multi-turn dialogue corpus. Add a separate dialogue stage only after genuine
reviewed dialogue data becomes available.

Run on one or more GPUs with the same entry point:

```bash
NUM_GPUS=1 bash recipes/qwen3-1.7b-full/run.sh /absolute/path/to/verified/model

CUDA_VISIBLE_DEVICES=0,1,2 NUM_GPUS=3 \
  bash recipes/qwen3-1.7b-full/run.sh /absolute/path/to/verified/model
```

The runner refuses to overlap an existing GPU job. A completed stage is reused
only when its top-level `adapter_model.safetensors` exists; a partial output
directory stops the pipeline for manual inspection.
