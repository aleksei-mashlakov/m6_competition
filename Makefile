.PHONY: help create-venv isort black mypy test local-ci doc
.DEFAULT_GOAL := help
VENV=venv
PYTHON_BIN=$(VENV)/bin
SRC=src
NOTEBOOKS=notebooks

create-venv:  ## Create virtual environment
	rm -rf venv
	python3 -m venv $(VENV)
	$(PYTHON_BIN)/pip3 install --upgrade pip
	$(PYTHON_BIN)/pip3 install -r ./environment/requirements.txt

create-conda:
	conda create -f ./environment/conda.yml

isort:  ## Check formatting with isort
	$(PYTHON_BIN)/isort $(SRC) $(NOTEBOOKS) --check-only

black:  ## Check formatting with black
	$(PYTHON_BIN)/black $(SRC) $(NOTEBOOKS) --check

mypy:  ## Check typing with mypy
	$(PYTHON_BIN)/mypy $(SRC) --namespace-packages

test:  ## Run unit tests
	echo "Testing ..."

notebook-clear-output:  ## Clear jupyter notebook output
	$(PYTHON_BIN)/jupyter nbconvert \
	    --NbConvertApp.use_output_suffix=False \
	    --NbConvertApp.export_format=notebook \
	    --FilesWriter.build_directory= \
	    --ClearOutputPreprocessor.enabled=True \
	    --ClearMetadataPreprocessor.clear_cell_metadata=True \
		notebooks/*.ipynb

local-ci: isort black mypy test  ## Local CI

run-notebook: ## Launch notebook in background 
	jupyter notebook --no-browser &!

setup-pre-commit-hook:  ## Setup git pre-commit hook for local CI automation
	printf '#!/bin/sh\nmake local-ci || (echo "\nINFO: Use -n flag to skip CI pipeline" && false)' > .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit

data:
	mkdir -p data

data/training.csv: data
	curl https://filesamples.com/samples/document/csv/sample1.csv > data/training.csv
	echo "Downloaded data/training.csv"

train: data/training.csv
	python train.py --learning-rate 0.0001 --dataset data/training.csv


help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'