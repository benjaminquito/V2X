.PHONY: install test lint smoke reproduce

install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

smoke:
	python -m v2x_risk.smoke --config configs/smoke.yaml

reproduce:
	./scripts/reproduce.sh configs/default.yaml
