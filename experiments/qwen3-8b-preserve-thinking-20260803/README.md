# Qwen3-8B Nuosu capability-preserving adaptation

This experiment tests whether a small low-rank update can add Standard Liangshan
Yi mappings without erasing the post-trained Qwen3-8B reasoning policy.

The key control is the base model. Use `Qwen/Qwen3-8B`, not
`Qwen/Qwen3-8B-Base`: the former includes Qwen3 post-training and hybrid thinking
mode, while the latter does not. Every Nuosu SFT example is transformed into a
conversational prompt/completion pair and the final user message receives
`/no_think`. This makes the short translation target conditional on an explicit
non-thinking request instead of teaching the model to suppress thinking globally.
The official upstream chat template is retained.

## Experimental design

- `M0`: frozen official post-trained Qwen3-8B;
- `M1`: archived Base-derived translation adapter, negative control;
- `M2`: M0 plus one low-learning-rate Nuosu CPT epoch;
- `M3`: one contextual-`/no_think` completion-only SFT epoch, initialized from
  M2 only if the post-CPT reasoning canary retains at least 95% of M0; otherwise
  initialized directly from M0.

The primary preservation endpoint is the reasoning composite retention ratio
`score(M3) / score(M0)`. The preregistered non-inferiority threshold is 0.95.
The quick canary is only an operational gate; paper claims must use fixed public
benchmark revisions and bootstrap confidence intervals. NuosuBench overlaps are
always disclosed and are never described as an uncontaminated test. Final Nuosu
quality requires blinded native-speaker ratings.

The CPT-to-SFT transition is therefore conditional. A post-CPT canary result
below 0.95 stops adapter propagation; it does not stop the experiment or permit
retuning the threshold after seeing the result. This safety rule limits
reasoning drift while keeping the eight-item canary separate from formal model
selection and publication claims.

## Training stages

```bash
CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc_per_node=2 \
  scripts/train.py \
  --config configs/cpt_qwen3_8b_qlora.yaml \
  --base-model /absolute/path/to/Qwen3-8B-b968826d

CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc_per_node=2 \
  scripts/train.py \
  --config configs/sft_qwen3_8b_qlora.yaml \
  --base-model /absolute/path/to/Qwen3-8B-b968826d
```

Run `scripts/download_training_corpus.py` first. Record code SHA, dependency
lock, hardware inventory, data hashes, and full command lines in the experiment
artifact directory before the baseline evaluation.

Create the machine-readable preflight record with:

```bash
python scripts/capture_run_metadata.py \
  --experiment-id qwen3-8b-nuosu-preserve-thinking-20260803 \
  --output artifacts/experiments/qwen3-8b-preserve-thinking-20260803/preflight.json \
  --input experiments/qwen3-8b-preserve-thinking-20260803/experiment.yaml \
  --input /absolute/path/to/Qwen3-8B-b968826d/snapshot_manifest.json \
  --input ../nuosu-corpus/data/hf/nuosu-corpus/ready_cpt.jsonl \
  --input ../nuosu-corpus/data/hf/nuosu-corpus/ready_sft.jsonl
```
