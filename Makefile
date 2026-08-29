.PHONY: setup data train evaluate serve demo test lint typecheck security terraform-validate clean
PYTHON ?= python
setup:
	$(PYTHON) -m pip install -e ".[dev]"
data:
	$(PYTHON) -m decision_platform.cli generate-data
train:
	$(PYTHON) -m decision_platform.cli train
evaluate:
	$(PYTHON) -m decision_platform.cli evaluate
serve:
	$(PYTHON) -m uvicorn decision_platform.serving.api:app --host 0.0.0.0 --port 8000
demo:
	$(PYTHON) -m decision_platform.cli demo
test:
	$(PYTHON) -m pytest
lint:
	$(PYTHON) -m ruff check src tests
typecheck:
	$(PYTHON) -m mypy src
security:
	$(PYTHON) -m bandit -q -r src
terraform-validate:
	terraform -chdir=infrastructure/terraform/environments/dev init -backend=false
	terraform -chdir=infrastructure/terraform/environments/dev validate
clean:
	$(PYTHON) -m decision_platform.cli clean

