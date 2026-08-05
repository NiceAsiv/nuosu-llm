# Qwen3-1.7B Nuosu MT pipeline

This is the canonical end-to-end recipe for the fast Nuosu machine-translation model.
One command performs:

1. deterministic target-only MT projection with a rejection audit;
2. tokenizer expansion with all 1,165 Standard Yi syllables and 55 Yi radicals;
3. CPT token adaptation and MT SFT, with automatic checkpoint recovery;
4. deterministic base and final-model generation on the fixed research split;
5. chrF2/CER/exact-match scoring, adapter inspection, provenance and SHA-256 manifests.

On the three-RTX-3090 research server:

```bash
cd /home/xjtuoss/nuosu-project/nuosu-llm-20260803
NUM_GPUS=3 bash recipes/qwen3-1.7b-mt/run.sh
```

The default verified base and corpus paths match that server. Override them portably:

```bash
NUOSU_BASE_MODEL=/models/Qwen3-1.7B-Base \
NUOSU_CORPUS_DIR=/data/nuosu-corpus-v2026.08.04 \
NUM_GPUS=2 \
bash recipes/qwen3-1.7b-mt/run.sh
```

Completed stages are reused. If a stage stopped after producing a checkpoint, rerunning the
same command resumes the latest numeric `checkpoint-*`. A partial output without a checkpoint
is never deleted automatically.

The SDPA recipe deliberately keeps packing disabled. TRL's padding-free packed batches require
a supported FlashAttention implementation; using them with SDPA can leak attention across packed
examples and is therefore not accepted by this pipeline.

Set `EVAL_BASELINE=0` only for a recovery run where the immutable base evaluation is already
available. The trained model is not considered complete until the final generation file,
metrics and `COMPLETED` marker exist in `artifacts/pipeline/qwen3-1.7b-mt/<run-id>/`.
