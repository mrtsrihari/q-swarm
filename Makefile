.PHONY: install test lint format demo dashboard benchmark clean help

help:
	@echo "Q-Swarm RL: Available commands"
	@echo "  make install     - Install development dependencies in editable mode"
	@echo "  make test        - Run automated pytest suite"
	@echo "  make lint        - Run ruff linter check"
	@echo "  make format      - Run ruff formatter and fixer"
	@echo "  make demo        - Run quick heuristic simulation demo"
	@echo "  make dashboard   - Launch Streamlit interactive dashboard"
	@echo "  make benchmark   - Run smoke benchmark comparison"
	@echo "  make clean       - Remove cached files and build artifacts"

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests -v

lint:
	ruff check .

format:
	ruff check --fix .
	ruff format .

demo:
	python scripts/run_demo.py --policy heuristic --scenario rescue_maze --seed 42

dashboard:
	streamlit run src/qswarm/dashboard/app.py

benchmark:
	python scripts/run_benchmark.py --config configs/experiments/benchmark.yaml --smoke

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.pytest_cache')]"
