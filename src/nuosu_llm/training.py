from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import load_config, with_overrides
from .tokenizer_expansion import (
    plan_tokenizer_expansion,
    resize_and_initialize_embeddings,
    write_expansion_manifest,
)

QWEN3_ASSISTANT_MASK_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' }}"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}{{ message['content'] + '<|endoftext|>' }}{% endgeneration %}"
    "{% else %}{{ message['content'] + '<|im_end|>\\n' }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def ensure_assistant_mask_chat_template(
    tokenizer: Any, assistant_only_loss: bool
) -> bool:
    """Install a Qwen-compatible template with generation spans when needed."""
    if not assistant_only_loss:
        return False
    template = tokenizer.chat_template or ""
    if "{% generation %}" in template and "{% endgeneration %}" in template:
        return False
    tokenizer.chat_template = QWEN3_ASSISTANT_MASK_CHAT_TEMPLATE
    return True


def _dtype(torch: Any, name: str) -> Any:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise ValueError(f"不支持的 compute_dtype: {name}") from error


def initialize_distributed_runtime(torch: Any, local_rank: int, world_size: int) -> None:
    """Bind each torchrun rank to one GPU before Accelerate creates barriers."""
    if not torch.cuda.is_available():
        return
    torch.cuda.set_device(local_rank)
    if (
        world_size > 1
        and torch.distributed.is_available()
        and not torch.distributed.is_initialized()
    ):
        torch.distributed.init_process_group(
            backend="nccl",
            device_id=torch.device("cuda", local_rank),
        )


def training_sampling_strategy_name(group_by_length: bool) -> str:
    """Resolve the sampler label without relying on optional TRL attributes."""
    return "group_by_length" if group_by_length else "random"


def to_prompt_completion(
    messages: list[dict[str, Any]], *, append_no_think: bool
) -> dict[str, list[dict[str, Any]]]:
    """Preserve Qwen3's official thinking template while masking prompt loss."""
    copied = deepcopy(messages)
    if not copied or copied[-1].get("role") != "assistant":
        raise ValueError("prompt/completion SFT 要求最后一条消息为 assistant")
    if append_no_think:
        for message in reversed(copied[:-1]):
            if message.get("role") != "user":
                continue
            content = str(message.get("content") or "").rstrip()
            if "/no_think" not in content:
                message["content"] = f"{content}\n\n/no_think"
            break
        else:
            raise ValueError("prompt/completion SFT 缺少 user 消息")
    return {"prompt": copied[:-1], "completion": copied[-1:]}


def load_stage_rows(
    path: str | Path,
    stage: str,
    *,
    prompt_completion: bool = False,
    append_no_think: bool = False,
    repeat_by_task: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Load only trainer-facing fields so heterogeneous metadata cannot break Arrow."""
    if stage not in {"cpt", "sft"}:
        raise ValueError(f"不支持的训练阶段: {stage}")
    field = "text" if stage == "cpt" else "messages"
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: JSON 无效: {error}") from error
            value = record.get(field) if isinstance(record, dict) else None
            if stage == "cpt" and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{path}:{line_number}: CPT 记录缺少非空 text")
            if stage == "sft" and (not isinstance(value, list) or not value):
                raise ValueError(f"{path}:{line_number}: SFT 记录缺少非空 messages")
            repeat = 1
            if stage == "sft" and repeat_by_task:
                task = str(record.get("task") or "")
                repeat = repeat_by_task.get(task, 1)
                if not isinstance(repeat, int) or repeat < 1:
                    raise ValueError(
                        f"{path}:{line_number}: task {task!r} 的重复次数必须是正整数"
                    )
            trainer_row = (
                to_prompt_completion(value, append_no_think=append_no_think)
                if stage == "sft" and prompt_completion
                else {field: value}
            )
            rows.extend(deepcopy(trainer_row) for _ in range(repeat))
    if not rows:
        raise ValueError(f"{path}: 没有可用的 {stage.upper()} 记录")
    return rows


def require_verified_base_model(config: dict[str, Any]) -> Path:
    """Require an explicit local model recovered by the integrity workflow."""
    base_model = Path(str(config["base_model"])).expanduser()
    if not base_model.is_dir():
        raise FileNotFoundError(
            "基础模型必须是已校验的本地目录；请先运行 "
            "scripts/model/README.md 完成下载与验证，并用 --base-model 覆盖配置。"
        )
    verification_marker = base_model / "VERIFIED.sha256"
    if not verification_marker.is_file():
        raise FileNotFoundError(
            f"基础模型缺少校验标记: {verification_marker}。"
            "禁止直接训练未通过实际 SHA-256 校验的模型。"
        )
    return base_model


def run_training(config: dict[str, Any]) -> None:
    require_verified_base_model(config)

    import torch
    from datasets import Dataset, DatasetDict
    from peft import (
        LoraConfig,
        PeftModel,
        prepare_model_for_kbit_training,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    class CompactEmbeddingSFTTrainer(SFTTrainer):
        """Prevent PEFT from redundantly saving full resized embedding matrices."""

        def _save(
            self,
            output_dir: str | None = None,
            state_dict: dict[str, Any] | None = None,
        ) -> None:
            if expansion_plan is None:
                super()._save(output_dir=output_dir, state_dict=state_dict)
                return
            target_dir = output_dir or self.args.output_dir
            os.makedirs(target_dir, exist_ok=True)
            unwrapped = self.accelerator.unwrap_model(
                self.model, keep_torch_compile=False
            )
            unwrapped.save_pretrained(
                target_dir,
                state_dict=state_dict,
                save_embedding_layers=False,
            )
            if self.processing_class is not None:
                self.processing_class.save_pretrained(target_dir)
            torch.save(self.args, os.path.join(target_dir, "training_args.bin"))

    stage = config["stage"]
    quant = config.get("quantization", {})
    training = config["training"]
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    gradient_checkpointing = bool(training.get("gradient_checkpointing", True))
    assistant_only_loss = bool(training.get("assistant_only_loss", False))
    completion_only_loss = bool(training.get("completion_only_loss", False))
    prompt_completion = bool(training.get("prompt_completion", False))
    append_no_think = bool(training.get("append_no_think", False))
    data_sampling = config.get("data_sampling", {})
    repeat_by_task = data_sampling.get("repeat_by_task", {})
    if not isinstance(repeat_by_task, dict):
        raise ValueError("data_sampling.repeat_by_task 必须是 task 到正整数的映射")
    initialize_distributed_runtime(torch, local_rank, world_size)

    quantization_config = None
    if quant.get("enabled", True):
        if quant.get("bits", 4) != 4:
            raise ValueError("当前实现只支持 4-bit QLoRA")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant.get("quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant.get("double_quant", True),
            bnb_4bit_compute_dtype=_dtype(torch, quant.get("compute_dtype", "bfloat16")),
        )

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if stage == "sft":
        ensure_assistant_mask_chat_template(tokenizer, assistant_only_loss)
    expansion_config = config.get("tokenizer_expansion", {})
    expansion_plan = (
        plan_tokenizer_expansion(tokenizer, expansion_config)
        if expansion_config.get("enabled", False)
        else None
    )

    model_kwargs = {
        "quantization_config": quantization_config,
        "dtype": _dtype(torch, quant.get("compute_dtype", "bfloat16")),
        "device_map": {"": local_rank},
    }
    attn_implementation = training.get("attn_implementation")
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(config["base_model"], **model_kwargs)
    if expansion_plan is not None:
        resize_and_initialize_embeddings(
            model,
            tokenizer,
            expansion_plan,
            pad_to_multiple_of=int(expansion_config.get("pad_to_multiple_of", 64)),
        )
    model.config.use_cache = False

    init_adapter = config.get("init_adapter")
    peft_config = None
    if init_adapter:
        adapter_path = Path(init_adapter)
        if not adapter_path.exists():
            raise FileNotFoundError(f"初始 LoRA 适配器不存在: {adapter_path}")
        if quantization_config is not None:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=True,
        )

    loader_kwargs = {
        "prompt_completion": prompt_completion,
        "append_no_think": append_no_think,
    }
    train_rows = load_stage_rows(
        config["train_file"],
        stage,
        repeat_by_task=repeat_by_task,
        **loader_kwargs,
    )
    dataset_splits = {
        "train": Dataset.from_list(train_rows)
    }
    eval_path = config.get("eval_file")
    if eval_path and Path(eval_path).exists():
        dataset_splits["eval"] = Dataset.from_list(
            load_stage_rows(eval_path, stage, **loader_kwargs)
        )
    datasets = DatasetDict(dataset_splits)

    lora = config["lora"]
    if not init_adapter:
        peft_config = LoraConfig(
            r=int(lora.get("r", 32)),
            lora_alpha=int(lora.get("alpha", 64)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            target_modules=lora.get("target_modules", "all-linear"),
            bias=lora.get("bias", "none"),
            task_type="CAUSAL_LM",
            trainable_token_indices=(
                list(expansion_plan.added_token_ids)
                if expansion_plan is not None
                else None
            ),
            ensure_weight_tying=bool(expansion_plan is not None),
        )

    report_to = training.get("report_to", "none")
    report_targets = [] if report_to in {None, "none"} else [report_to]
    eval_enabled = "eval" in datasets
    load_best_model_at_end = bool(
        training.get("load_best_model_at_end", eval_enabled)
    )
    train_batch_size = int(training.get("per_device_train_batch_size", 1))
    accumulation_steps = int(training.get("gradient_accumulation_steps", 16))
    dataloader_workers = int(training.get("dataloader_num_workers", 0))
    group_by_length = bool(training.get("group_by_length", False))
    sft_kwargs = dict(
        output_dir=config["output_dir"],
        seed=int(config.get("seed", 42)),
        max_length=int(training.get("max_length", 2048)),
        packing=bool(training.get("packing", True)),
        packing_strategy=training.get("packing_strategy", "bfd"),
        padding_free=bool(training.get("padding_free", False)),
        eval_packing=training.get("eval_packing"),
        assistant_only_loss=assistant_only_loss,
        completion_only_loss=completion_only_loss,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=int(training.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if gradient_checkpointing else None
        ),
        learning_rate=float(training.get("learning_rate", 1e-4)),
        num_train_epochs=float(training.get("num_train_epochs", 1)),
        warmup_ratio=float(training.get("warmup_ratio", 0.03)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        lr_scheduler_type=training.get("lr_scheduler_type", "cosine"),
        logging_steps=int(training.get("logging_steps", 10)),
        eval_strategy=training.get(
            "eval_strategy", "steps" if eval_enabled else "no"
        ),
        eval_steps=int(training.get("eval_steps", 250)),
        save_strategy=training.get("save_strategy", "steps"),
        save_steps=int(training.get("save_steps", 250)),
        save_total_limit=int(training.get("save_total_limit", 3)),
        bf16=bool(training.get("bf16", True)),
        fp16=bool(training.get("fp16", False)),
        tf32=bool(training.get("tf32", False)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        ddp_find_unused_parameters=bool(
            training.get("ddp_find_unused_parameters", False)
        ),
        ddp_bucket_cap_mb=training.get("ddp_bucket_cap_mb"),
        ddp_broadcast_buffers=bool(training.get("ddp_broadcast_buffers", False)),
        dataloader_num_workers=dataloader_workers,
        dataloader_pin_memory=bool(training.get("dataloader_pin_memory", True)),
        dataloader_persistent_workers=bool(
            training.get("dataloader_persistent_workers", dataloader_workers > 0)
        ),
        optim=training.get("optim", "adamw_torch"),
        max_steps=int(training.get("max_steps", -1)),
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=(
            training.get("metric_for_best_model", "eval_loss")
            if load_best_model_at_end
            else None
        ),
        greater_is_better=(
            bool(training.get("greater_is_better", False))
            if load_best_model_at_end
            else None
        ),
        report_to=report_targets,
    )
    if stage == "cpt":
        sft_kwargs["dataset_text_field"] = "text"
    train_sampling_strategy = training_sampling_strategy_name(group_by_length)
    if dataloader_workers > 0:
        sft_kwargs["dataloader_prefetch_factor"] = int(
            training.get("dataloader_prefetch_factor", 2)
        )
    sft_args = SFTConfig(**sft_kwargs)

    if local_rank == 0:
        startup_metrics = {
            "world_size": world_size,
            "per_device_train_batch_size": train_batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "effective_global_batch_size": (
                world_size * train_batch_size * accumulation_steps
            ),
            "compute_dtype": quant.get("compute_dtype", "bfloat16"),
            "packing": sft_args.packing,
            "packing_strategy": sft_args.packing_strategy,
            "group_by_length": group_by_length,
            # TRL 0.29 does not expose this as an SFTConfig attribute when the
            # default random sampler is used.  Report the resolved local value
            # instead of coupling provenance logging to an optional attribute.
            "train_sampling_strategy": train_sampling_strategy,
            "assistant_only_loss": assistant_only_loss,
            "completion_only_loss": completion_only_loss,
            "prompt_completion": prompt_completion,
            "append_no_think": append_no_think,
            "train_rows_after_sampling": len(train_rows),
            "eval_rows": len(datasets.get("eval", [])),
            "repeat_by_task": repeat_by_task,
            "preserves_original_chat_template": not assistant_only_loss,
            "attention": attn_implementation or "model_default",
            "tokenizer_expansion": (
                expansion_plan.manifest() if expansion_plan is not None else None
            ),
        }
        print("NUOSU_TRAINING_TOPOLOGY=" + json.dumps(startup_metrics, sort_keys=True))

    trainer = CompactEmbeddingSFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("eval"),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    if training.get("sanitize_nonfinite_gradients", False):
        for parameter in trainer.model.parameters():
            if parameter.requires_grad:
                parameter.register_hook(
                    lambda gradient: gradient.nan_to_num(
                        nan=0.0, posinf=0.0, neginf=0.0
                    )
                )
    trainer.train(resume_from_checkpoint=training.get("resume_from_checkpoint"))
    trainer.save_model(config["output_dir"])
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(config["output_dir"])
        if expansion_plan is not None:
            write_expansion_manifest(
                Path(config["output_dir"]) / "tokenizer_expansion.json",
                expansion_plan,
            )
    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Nuosu CPT or SFT QLoRA adapter")
    parser.add_argument("--config", required=True, help="YAML 配置文件")
    parser.add_argument("--base-model", help="已通过 VERIFIED.sha256 门禁的本地模型目录")
    parser.add_argument("--train-file")
    parser.add_argument("--eval-file")
    parser.add_argument("--output-dir")
    parser.add_argument("--init-adapter")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from-checkpoint")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = with_overrides(
        load_config(args.config),
        base_model=args.base_model,
        train_file=args.train_file,
        eval_file=args.eval_file,
        output_dir=args.output_dir,
        init_adapter=args.init_adapter,
        max_steps=args.max_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    run_training(config)


if __name__ == "__main__":
    main()
