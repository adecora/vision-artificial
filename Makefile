QUARTO ?= quarto
QUARTO_PYTHON ?= .venv/bin/python
QUARTO_CMD = QUARTO_PYTHON=$(QUARTO_PYTHON) $(QUARTO)

.PHONY: html pdf preview slides slides-01 slides-02 clean check

html:
	$(QUARTO_CMD) render --to html

pdf:
	$(QUARTO_CMD) render --no-clean --to pdf

preview:
	$(QUARTO_CMD) preview --to html

check:
	$(QUARTO_CMD) check

clean:
	rm -rf httdocs .quarto _freeze
