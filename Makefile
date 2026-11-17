.PHONY: help install data train sample test lint dashboard clean

PY ?= python

help:       ## Show this help
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

install:    ## Install with dev and notebook extras
	$(PY) -m pip install -e ".[dev,notebook]"

data:       ## Simulate the benchmark market
	tempora data simulate

train:      ## Train, evaluate and export dashboard data (~1 min)
	tempora train

sample:     ## Write 100 synthetic 1-year price paths
	tempora sample --paths 100 --days 250

test:       ## Run tests
	$(PY) -m pytest

lint:       ## Lint
	ruff check src tests

dashboard:  ## Serve the dashboard at http://localhost:8000
	$(PY) -m http.server 8000 --bind 127.0.0.1 --directory dashboard

clean:      ## Remove caches
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
