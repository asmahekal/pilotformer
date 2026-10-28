.PHONY: install test lint smoke reproduce

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

smoke:
	pilotformer reproduce --config configs/smoke.yaml

reproduce:
	pilotformer reproduce --config configs/paper.yaml
