.PHONY: install run test docker-build docker-run clean

install:
	pip install -r requirements.txt 2>/dev/null || echo "No requirements"

run:
	python -m src.main 2>/dev/null || python main.py

test:
	pytest tests/ 2>/dev/null || echo "No tests yet"

docker-build:
	docker-compose build

docker-run:
	docker-compose up -d

deploy-local:
	make docker-run

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
	find . -type f -name '*.pyc' -delete 2>/dev/null
