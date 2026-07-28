from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a training configuration is incomplete or inconsistent."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError("配置文件根节点必须是 YAML 对象")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {"stage", "base_model", "train_file", "output_dir", "lora", "training"}
    missing = sorted(required.difference(config))
    if missing:
        raise ConfigError(f"缺少配置项: {', '.join(missing)}")
    if config["stage"] not in {"cpt", "sft"}:
        raise ConfigError("stage 必须是 cpt 或 sft")
    if config["stage"] == "sft" and not config["training"].get("assistant_only_loss", False):
        raise ConfigError("SFT 默认要求 assistant_only_loss: true")


def with_overrides(
    config: dict[str, Any],
    *,
    train_file: str | None = None,
    eval_file: str | None = None,
    output_dir: str | None = None,
    init_adapter: str | None = None,
    max_steps: int | None = None,
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    updated = deepcopy(config)
    if train_file:
        updated["train_file"] = train_file
    if eval_file:
        updated["eval_file"] = eval_file
    if output_dir:
        updated["output_dir"] = output_dir
    if init_adapter:
        updated["init_adapter"] = init_adapter
    if max_steps is not None:
        updated["training"]["max_steps"] = max_steps
    if resume_from_checkpoint:
        updated["training"]["resume_from_checkpoint"] = resume_from_checkpoint
    return updated
