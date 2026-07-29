from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


def inspect_adapter(path: Path) -> dict[str, Any]:
    weights_path = path / "adapter_model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"adapter weights not found: {weights_path}")

    tensors = 0
    elements = 0
    finite_elements = 0
    zero_elements = 0
    sum_abs = 0.0
    sum_squares = 0.0
    max_abs = 0.0

    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name).detach().float()
            finite = torch.isfinite(tensor)
            finite_tensor = tensor[finite]
            tensors += 1
            elements += tensor.numel()
            finite_elements += finite_tensor.numel()
            zero_elements += int((finite_tensor == 0).sum().item())
            if finite_tensor.numel():
                absolute = finite_tensor.abs()
                sum_abs += float(absolute.sum().item())
                sum_squares += float((finite_tensor * finite_tensor).sum().item())
                max_abs = max(max_abs, float(absolute.max().item()))

    return {
        "adapter": str(path),
        "weights": str(weights_path),
        "tensors": tensors,
        "elements": elements,
        "nonfinite_elements": elements - finite_elements,
        "zero_fraction": zero_elements / max(finite_elements, 1),
        "mean_abs": sum_abs / max(finite_elements, 1),
        "rms": math.sqrt(sum_squares / max(finite_elements, 1)),
        "max_abs": max_abs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check LoRA adapters for numerical damage")
    parser.add_argument("adapters", nargs="+", type=Path)
    parser.add_argument("--output")
    args = parser.parse_args()

    payload = {"adapters": [inspect_adapter(path) for path in args.adapters]}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
