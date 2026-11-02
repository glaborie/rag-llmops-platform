PYTHON ?= python
PORT ?= 8000

.PHONY: help install install-ml test lint fmt ingest evaluate monitor loop serve features pipeline reset docker-up docker-down clean

help:
	@echo "ragllmops-platform"
	@echo ""
	@echo "  make install     install the core dependencies (light)"
	@echo "  make install-ml  install the optional heavy backends"
	@echo "  make test        run the test suite"
	@echo "  make lint        run ruff"
	@echo "  make ingest      build the retrieval index from the corpus"
	@echo "  make evaluate    run the golden set and the quality gate"
	@echo "  make monitor     drift report over the request log"
	@echo "  make loop        run the closed loop (detect -> gate -> promote)"
	@echo "  make serve       run the API on port $(PORT)"
	@echo "  make features    materialise features and check offline/online parity"
	@echo "  make pipeline    run the platform DAG once"
	@echo "  make reset       clear the local state"
	@echo "  make docker-up   start the full local stack"
	@echo "  make docker-down stop the full local stack"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install --no-deps -e .

install-ml: install
	$(PYTHON) -m pip install -r requirements-ml.txt

test:
	$(PYTHON) -m pytest tests -q

lint:
	$(PYTHON) -m ruff check --no-cache src tests

fmt:
	$(PYTHON) -m ruff format --no-cache src tests
	$(PYTHON) -m ruff check --no-cache --fix src tests

ingest:
	$(PYTHON) -m ragllmops.cli ingest

evaluate:
	$(PYTHON) -m ragllmops.cli evaluate

monitor:
	$(PYTHON) -m ragllmops.cli monitor

loop:
	$(PYTHON) -m ragllmops.cli loop

serve:
	$(PYTHON) -m ragllmops.cli serve --port $(PORT)

features:
	$(PYTHON) -m ragllmops.cli features

pipeline:
	$(PYTHON) -m ragllmops.cli pipeline run

reset:
	$(PYTHON) -m ragllmops.cli reset

docker-up:
	docker compose up -d --build
	@echo "API      : http://localhost:$(PORT)/docs"
	@echo "MLflow   : http://localhost:5000"
	@echo "Qdrant   : http://localhost:6333/dashboard"
	@echo "Promethe: http://localhost:9090"

docker-down:
	docker compose down

clean: reset
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
