from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .config import load_config, with_overrides


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


def run_training(config: dict[str, Any]) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    stage = config["stage"]
    quant = config.get("quantization", {})
    training = config["training"]
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

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

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=quantization_config,
        torch_dtype=_dtype(torch, quant.get("compute_dtype", "bfloat16")),
        device_map={"": local_rank},
    )
    model.config.use_cache = False

    data_files = {"train": config["train_file"]}
    eval_path = config.get("eval_file")
    if eval_path and Path(eval_path).exists():
        data_files["eval"] = eval_path
    datasets = load_dataset("json", data_files=data_files)

    lora = config["lora"]
    peft_config = LoraConfig(
        r=int(lora.get("r", 32)),
        lora_alpha=int(lora.get("alpha", 64)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=lora.get("target_modules", "all-linear"),
        bias=lora.get("bias", "none"),
        task_type="CAUSAL_LM",
    )

    report_to = training.get("report_to", "none")
    report_targets = [] if report_to in {None, "none"} else [report_to]
    eval_enabled = "eval" in datasets

    sft_args = SFTConfig(
        output_dir=config["output_dir"],
        seed=int(config.get("seed", 42)),
        max_length=int(training.get("max_length", 2048)),
        packing=bool(training.get("packing", True)),
        assistant_only_loss=bool(training.get("assistant_only_loss", False)),
        per_device_train_batch_size=int(training.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(training.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 16)),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        learning_rate=float(training.get("learning_rate", 1e-4)),
        num_train_epochs=float(training.get("num_train_epochs", 1)),
        warmup_ratio=float(training.get("warmup_ratio", 0.03)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        lr_scheduler_type=training.get("lr_scheduler_type", "cosine"),
        logging_steps=int(training.get("logging_steps", 10)),
        eval_strategy="steps" if eval_enabled else "no",
        eval_steps=int(training.get("eval_steps", 250)),
        save_strategy="steps",
        save_steps=int(training.get("save_steps", 250)),
        save_total_limit=int(training.get("save_total_limit", 3)),
        bf16=bool(training.get("bf16", True)),
        fp16=bool(training.get("fp16", False)),
        report_to=report_targets,
        dataset_text_field="text" if stage == "cpt" else None,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("eval"),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=training.get("resume_from_checkpoint"))
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Nuosu CPT or SFT QLoRA adapter")
    parser.add_argument("--config", required=True, help="YAML 配置文件")
    parser.add_argument("--train-file")
    parser.add_argument("--eval-file")
    parser.add_argument("--output-dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = with_overrides(
        load_config(args.config),
        train_file=args.train_file,
        eval_file=args.eval_file,
        output_dir=args.output_dir,
    )
    run_training(config)


if __name__ == "__main__":
    main()

