.PHONY: data train evaluate test app clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

install-dev: ## Install dev + test dependencies
	pip install -r requirements.txt -r requirements-dev.txt

data: ## Run data curation pipeline (ChEMBL → cleaned parquet)
	python scripts/01_data_curation.py

train: ## Train and benchmark models
	python scripts/02_model_training.py

evaluate: ## Generate evaluation figures and metrics
	python scripts/03_model_evaluation.py

test: ## Run unit tests
	python -m pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	python -m pytest tests/ -v --cov=src --cov-report=term-missing

app: ## Launch Streamlit application
	streamlit run app.py

pipeline: data train evaluate ## Run full pipeline (data → train → evaluate)

clean: ## Remove generated artifacts
	rm -rf data/raw data/processed data/splits
	rm -rf models/trained models/figures
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
