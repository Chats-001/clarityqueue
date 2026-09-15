# ClarityQueue

ClarityQueue is a small evidence-grounded support intelligence system. It reads an incoming case,
predicts the right support team and urgency, retrieves a relevant runbook, and produces a cited
next-step response only when the evidence is strong enough. The project combines interpretable ML,
retrieval evaluation, analytics, FastAPI, SQLite/SQL, and a results-first Streamlit dashboard.

The useful question is not “Can an AI answer everything?” It is: **Which cases can the system help
with reliably, and which should stay with a person?**

## What the benchmark showed

| Outcome | Held-out result | Why it matters |
|---|---:|---|
| Route accuracy | **85.0%** | Most cases reach the intended support team |
| Route macro-F1 | **85.2%** | Quality is measured across all five teams, not just the largest |
| Priority macro-F1 | **72.8%** | Urgent cases are easier than the intentionally ambiguous low/medium split |
| Runbook hit@3 | **100.0%** | The expected evidence appears within three results in the controlled KB |
| High-confidence automation candidates | **74.2%** | A quarter of cases remain available for human review |
| Out-of-domain abstention | **87.5%** | Unrelated questions usually produce “not enough evidence” |

These are reproducible results on an original synthetic benchmark of 360 tickets and 15 authored
runbooks. Five complete issue families—one per route—are excluded from training and used only for
evaluation. They demonstrate the methodology, not expected performance on real customer traffic.

## Product flow

```text
Support case
    |
    +--> TF-IDF classifiers --> route + priority + visible token signals
    |
    +--> evidence index --> ranked runbooks --> threshold check
                                              |             |
                                           strong          weak
                                              |             |
                                      cited guidance    human review
                                              |
                                      SQLite event metrics
                                              |
                                      dashboard + SQL
```

## Why this is RAG

The response layer retrieves relevant operational knowledge and composes guidance exclusively from
the selected source steps. Every answered response includes an article citation. There is no hosted
LLM dependency: that keeps the project cheap, deterministic, and honest about what is being tested.
The same evaluation harness could later compare embedding or LLM-based components without changing
the API contract.

## Run it

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
python -m scripts.generate_data
python -m scripts.train
pytest
```

Start the results dashboard and API in separate terminals:

```bash
streamlit run dashboard/app.py
uvicorn clarityqueue.api.main:app --reload
```

The API documentation is available at `http://127.0.0.1:8000/docs`.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Artifact readiness and version |
| `POST` | `/v1/triage` | Route, priority, confidence, and token signals |
| `POST` | `/v1/retrieve` | Ranked runbook evidence |
| `POST` | `/v1/answer` | Cited guidance or an explicit abstention |
| `GET` | `/v1/metrics` | Frozen held-out benchmark results |

Example:

```bash
curl -X POST http://127.0.0.1:8000/v1/answer \
  -H 'Content-Type: application/json' \
  -d '{"title":"Webhook signature rejected","description":"Production reports an invalid HMAC."}'
```

Requests reject blank text, unexpected fields, and oversized payloads. Event analytics retain route,
priority, confidence, answer status, and latency; raw ticket text is deliberately not logged.

## Dashboard

The landing page is written for someone reviewing outcomes, not model internals. It includes:

- four plain-English KPI cards;
- per-team precision, recall, and F1;
- a confusion matrix that exposes specific routing mistakes;
- confidence distributions for correct and incorrect predictions;
- an answer coverage-versus-quality curve;
- a live case explorer with citations and abstention behavior.

## Repository guide

```text
clarityqueue/       ML, retrieval, evaluation, API, and storage code
scripts/            seeded data generation and training workflow
data/source/        original knowledge base
artifacts/          generated model and benchmark outputs
dashboard/          Streamlit results experience
sql/                operational analysis queries
tests/              38 behavior-focused tests
docs/               methodology, limitations, and generated findings
```

## Design choices

- **TF-IDF + logistic regression:** fast, inspectable, and appropriate for a small benchmark.
- **Issue-family holdout:** avoids flattering results from near-duplicate generated templates.
- **Separate route and priority models:** their failure modes and business meaning differ.
- **Abstention:** coverage is reported alongside correctness; weak evidence is not hidden.
- **No raw-text logs:** the analytics example avoids retaining ticket content.

## Limitations

The tickets and runbooks are synthetic, English-only, and much cleaner than real support data.
Retrieval runs over only 15 articles. Confidence is not calibrated for production automation, and no
human feedback loop is included. Before real use, evaluate on representative labeled cases, define
route-specific thresholds, test multilingual and adversarial inputs, and review privacy obligations.
See [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

## Originality

The ResilientNet brief inspired the emphasis on held-out evaluation, transparent risk, dashboards,
SQL, and human-readable findings. ClarityQueue’s domain, data, architecture, code, API, models, and
evaluation design are original and independently implemented.

