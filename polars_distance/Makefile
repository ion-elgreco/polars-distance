
SHELL=/bin/bash

venv:  ## Set up virtual environment
	python3 -m venv venv
	venv/bin/pip install -r requirements.txt

install: venv ## Install polars-distance in develop mode
	unset CONDA_PREFIX && \
	source venv/bin/activate && maturin develop

install-release: venv ## Install polars-distance in release mode
	unset CONDA_PREFIX && \
	source venv/bin/activate && maturin develop --release

lint: venv ## Run linter
	unset CONDA_PREFIX && \
	source venv/bin/activate && ruff check --fix . && cargo check

format: venv ## Run formatters
	unset CONDA_PREFIX && \
	source venv/bin/activate && ruff format . && cargo fmt

clean: ## Clean venv and clean cargo
	-@rm -r venv
	-@cd polars_distance && cargo clean

.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort