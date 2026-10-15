.PHONY: help install test test-cov lint format run docker-up docker-down clean

help:
	@echo "Available commands:"
	@echo "  make install    - Install Python dependencies"
	@echo "  make test       - Run pytest test suite"
	@echo "  make test-cov   - Run tests with coverage report"
	@echo "  make lint       - Run ruff linter check"
	@echo "  make format     - Run ruff formatter"
	@echo "  make run        - Run Streamlit application locally"
	@echo "  make docker-up  - Boot full containerized stack via Docker Compose"
	@echo "  make docker-down- Stop running Docker Compose stack"
	@echo "  make clean      - Clean cache and temporary files"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install ruff pytest pytest-cov

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=engine --cov-report=term-missing

lint:
	ruff check .

format:
	ruff format .

run:
	streamlit run app.py

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
