.PHONY: setup train demo api test

setup: .venv/bin/python

.venv/bin/python:
	python3 -m venv .venv
	.venv/bin/python -m pip install -e '.[dev]'

train: setup
	.venv/bin/python -m scripts.generate_data
	.venv/bin/python -m scripts.train

demo: setup
	.venv/bin/streamlit run dashboard/app.py

api: setup
	.venv/bin/uvicorn clarityqueue.api.main:app --reload

test: setup
	.venv/bin/ruff check .
	.venv/bin/pytest
