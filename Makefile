.PHONY: help install test test-verbose test-cov clean

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	uv sync

install-dev:  ## Install dependencies including dev dependencies
	uv sync --all-extras

test:  ## Run all tests
	uv run pytest

test-verbose:  ## Run tests with verbose output
	uv run pytest -v

test-cov:  ## Run tests with coverage report
	uv run pytest --cov=.github/scripts --cov-report=term --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

test-watch:  ## Run tests in watch mode (requires pytest-watch)
	uv run pytest-watch

clean:  ## Clean up generated files
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf **/__pycache__
	rm -rf **/*.pyc
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
