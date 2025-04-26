install:
	pip install --upgrade pip
	pip install .

lint:
	flake8 main/
	black --check main/

format:
	black main/

test:
	pytest tests/ -m "unit"

performance-test:
	pytest tests/ -m "performance"

coverage:
	pytest --cov=main --cov-report=term-missing --cov-report=xml tests/

serve:
	uvicorn app.main:app --host 0.0.0.0 --port 8000
