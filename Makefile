.PHONY: help test examples ir demo clean lint install

PYTHON ?= python
export PYTHONPATH := src

EXAMPLES := $(wildcard examples/*.lum)

help: ## Show this help message
	@echo "lumenc - available targets:"
	@echo "  make test      Run the unit test suite"
	@echo "  make ir        Compile examples/*.lum to LLVM IR (.ll)"
	@echo "  make examples  Compile and run every example via --run-ir"
	@echo "  make demo      Compile and run examples/hello.lum with stats"
	@echo "  make lint      Run pyflakes over src and tests"
	@echo "  make install   Install lumenc in editable mode"
	@echo "  make clean     Remove generated .ll/.o/.exe and __pycache__"

test: ## Run the unit test suite
	$(PYTHON) -m unittest discover -s tests -v

ir: ## Compile all examples to LLVM IR
	@for f in $(EXAMPLES); do \
		echo "$(PYTHON) -m lumen $$f -o $${f%.lum}.ll"; \
		$(PYTHON) -m lumen $$f -o $${f%.lum}.ll || exit 1; \
	done

examples: ## Compile and run all examples through the IR interpreter
	@for f in $(EXAMPLES); do \
		echo "== $$f"; \
		$(PYTHON) -m lumen $$f --run-ir || exit 1; \
	done

demo: ## Run the hello world demo
	$(PYTHON) -m lumen examples/hello.lum --run-ir --stats

lint: ## Run pyflakes (optional dependency)
	-$(PYTHON) -m pyflakes src tests

install: ## Install lumenc in editable mode
	pip install -e .

clean: ## Remove generated artifacts
	rm -f *.ll *.o *.exe
	rm -rf __pycache__
	find . -type d -name __pycache__ -prune -exec rm -rf {} +