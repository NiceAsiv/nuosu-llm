.DEFAULT_GOAL := help

PYTHON ?= python
MODEL_DIR ?=

.PHONY: help install test lint model-recovery-plan model-recover model-smoke profile

help:
	@echo "Nuosu LLM"
	@echo "  make install               Install train + development dependencies"
	@echo "  make test                  Run unit tests"
	@echo "  make lint                  Run Ruff"
	@echo "  make model-recovery-plan   Dry-run model quarantine and recovery"
	@echo "  make model-recover         Execute quarantine, verified download and smoke test"
	@echo "  make model-smoke MODEL_DIR=/abs/path"

install:
	$(PYTHON) -m pip install -e ".[train,dev]"

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

model-recovery-plan:
	PYTHON_BIN="$(PYTHON)" bash scripts/model/recover_qwen3_base.sh

model-recover:
	PYTHON_BIN="$(PYTHON)" bash scripts/model/recover_qwen3_base.sh --execute

model-smoke:
	@test -n "$(MODEL_DIR)" || (echo "MODEL_DIR is required" >&2; exit 2)
	$(PYTHON) scripts/model/smoke_test_base.py --model "$(MODEL_DIR)"

profile:
	$(PYTHON) scripts/profile_dataset.py --help
