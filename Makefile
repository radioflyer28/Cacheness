.PHONY: help setup-env up down logs clean test test-integration test-postgres test-s3

help:
	@echo "Cacheness Development Commands"
	@echo ""
	@echo "Local Test Environment:"
	@echo "  make setup-env          Setup local PostgreSQL and MinIO containers"
	@echo "  make up                 Start containers"
	@echo "  make down               Stop containers"
	@echo "  make clean              Stop containers and remove data"
	@echo "  make logs               Show container logs"
	@echo "  make logs-postgres      Show PostgreSQL logs"
	@echo "  make logs-minio         Show MinIO logs"
	@echo ""
	@echo "Testing:"
	@echo "  make test               Run all tests"
	@echo "  make test-integration   Run integration tests only"
	@echo "  make test-postgres      Run PostgreSQL tests"
	@echo "  make test-s3            Run S3 tests"
	@echo "  make test-quick         Run quick unit tests (no containers)"
	@echo ""
	@echo "Development:"
	@echo "  make install            Install package with all dependencies"
	@echo "  make lint               Run linters (ruff)"
	@echo "  make format             Format code (ruff)"
	@echo ""

setup-env:
	@python scripts/setup_local_env.py

up:
	docker-compose up -d
	@echo "Waiting for services to be healthy..."
	@sleep 10
	docker-compose ps

down:
	docker-compose down

clean:
	docker-compose down -v

logs:
	docker-compose logs -f

logs-postgres:
	docker-compose logs -f postgres

logs-minio:
	docker-compose logs -f minio

test:
	pytest tests/ -v

test-integration: up
	pytest tests/test_postgres_s3_integration.py tests/test_postgresql_integration.py tests/test_s3_integration.py -v

test-postgres: up
	pytest tests/test_postgresql_integration.py -v

test-s3: up
	pytest tests/test_s3_integration.py tests/test_s3_blob_backend.py -v

test-quick:
	pytest tests/ -v \
		--ignore=tests/test_postgres_s3_integration.py \
		--ignore=tests/test_postgresql_integration.py \
		--ignore=tests/test_s3_integration.py

install:
	pip install -e '.[dev,s3,postgresql,cloud]'

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check src/ tests/ --fix

.DEFAULT_GOAL := help
