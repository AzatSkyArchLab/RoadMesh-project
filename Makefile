.PHONY: install dev test lint format clean train api docker

# Installation
install:
	pip install -e .

dev:
	pip install -e ".[dev,train]"

# Development
lint:
	ruff check src/ scripts/
	mypy src/

format:
	black src/ scripts/
	ruff check --fix src/ scripts/

test:
	pytest tests/ -v

# Quick test
test-pipeline:
	python scripts/test_pipeline.py --area kremlin

# Training
train:
	python scripts/train.py --config configs/model/dlinknet_8gb.yaml

train-debug:
	python scripts/train.py --debug

# Dataset
dataset:
	python scripts/prepare_dataset.py \
		--vector-path data/raw/moscow_roads.geojson \
		--bbox 37.58,55.73,37.65,55.77 \
		--output-dir data/processed

# API server
api:
	uvicorn roadmesh.api.app:app --host 0.0.0.0 --port 8000 --reload

api-prod:
	uvicorn roadmesh.api.app:app --host 0.0.0.0 --port 8000 --workers 4

# Docker
docker-build:
	docker build -t roadmesh:latest -f docker/Dockerfile .

docker-run:
	docker-compose -f docker/docker-compose.yml up

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf build dist *.egg-info

clean-data:
	rm -rf data/cache/*
	rm -rf data/test_output/*

clean-all: clean clean-data
	rm -rf checkpoints/*
	rm -rf logs/*

# Help
help:
	@echo "Available commands:"
	@echo "  make install      - Install package"
	@echo "  make dev          - Install with dev dependencies"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make test-pipeline - Quick pipeline test"
	@echo "  make train        - Train model"
	@echo "  make train-debug  - Train in debug mode"
	@echo "  make dataset      - Prepare dataset"
	@echo "  make api          - Run API server (dev)"
	@echo "  make api-prod     - Run API server (prod)"
	@echo "  make docker-build - Build Docker image"
	@echo "  make clean        - Clean build artifacts"
