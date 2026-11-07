.PHONY: build up down logs build-data test lint format eval eval-live mcp clean

build:
	docker compose build

up:
	docker compose up

build-data:
	python scripts/build_olist_dataset.py
	python scripts/build_bitext_dataset.py
	python scripts/build_rescommons_dataset.py
	python scripts/build_v1rtucious_eval.py

down:
	docker compose down

logs:
	docker compose logs -f

test:
	python -m pytest tests/ -v

lint:
	ruff check app/ evaluation/ scripts/ tests/

format:
	ruff format app/ evaluation/ scripts/ tests/

eval:
	python -m evaluation.after_sales_bench --mode offline --suite core --control-mode full

eval-live:
	python -m evaluation.after_sales_bench --mode live --suite core --control-mode full
	python -m evaluation.after_sales_bench --mode live --suite llm_planner --control-mode full

mcp:
	python -m app.mcp_server

clean:
	docker compose down -v
