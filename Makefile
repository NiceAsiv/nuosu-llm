.PHONY: install test lint audit cpt sft

install:
	python -m pip install -e ".[train,dev]"

test:
	python -m pytest -q

lint:
	ruff check .

audit:
	python scripts/audit_tokenizers.py --help

cpt:
	python scripts/train.py --config configs/cpt_qwen3_8b_qlora.yaml

sft:
	python scripts/train.py --config configs/sft_qwen3_8b_qlora.yaml
