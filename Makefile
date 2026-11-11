QUARTO ?= quarto
QUARTO_PYTHON ?= .venv/bin/python
QUARTO_CMD = QUARTO_PYTHON=$(QUARTO_PYTHON) $(QUARTO)
CURRENT_DIR := $(shell pwd)

.PHONY: .uv
.uv:
	@uv -V || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.PHONY: jupyter
jupyter: .uv
	uv run jupyter lab

.PHONY: lint
lint:
	uvx ruff check
	uvx ty check

.PHONY: html
html: pdf
	$(QUARTO_CMD) render --to html

.PHONY: pdf
pdf:
	$(QUARTO_CMD) render --to pdf --clean --output-dir $(CURRENT_DIR)

.PHONY: preview
preview:
	$(QUARTO_CMD) preview --to html

.PHONY: check
check:
	$(QUARTO_CMD) check

.PHONY: clean
clean:
	rm -rf docs .quarto _freeze
