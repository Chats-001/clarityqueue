# ClarityQueue

ClarityQueue began as my first data science and AI project during a **Break Through Tech workshop
in December 2025**. I later rebuilt and expanded the idea into an explainable support decision
system that demonstrates not only how to train a model, but how to interpret its outcome, measure
uncertainty, connect predictions to an operational decision, and communicate limitations honestly.

The system reads an incoming case, predicts the right support team and urgency, retrieves a relevant
runbook, and produces a cited next-step response only when the evidence is strong enough. It combines
supervised and unsupervised ML, statistics, RAG evaluation, geospatial scenario analysis, FastAPI,
SQLite/SQL, automated tests, and a results-first Streamlit dashboard.

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

## What the outcome means

The model is promising as an **assistant**, not as a replacement for support staff. It could pre-sort
the 74.2% of cases that pass the current route-and-evidence gate, while leaving uncertain or unfamiliar
requests for human review. The error analysis also gives a concrete next step: Access and Data sync
cases that describe symptoms are sometimes mistaken for Performance problems, so better examples
that separate symptoms from root causes should come before more automation.

That conclusion is the most important result of the project. The dashboard is designed to show how I
moved from reporting one accuracy score to asking practical data-science questions about uncertainty,
model confidence, failure modes, retrieval safety, review workload, and responsible deployment.

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

The shortest path is one command from the repository folder:

```bash
make demo
```

Your browser will open the interactive dashboard at `http://localhost:8501`. The trained artifact
is committed, so you can explore the graphs immediately. To regenerate every result first, run
`make train`, followed by `make demo`.

The equivalent manual setup is:

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

The dashboard follows the project as a decision story rather than a collection of decorative charts:

- an outcome funnel showing how 120 unseen cases become 89 safe automation candidates;
- a normalized confusion matrix that turns mistakes into a data-collection recommendation;
- a simulated US operations map connecting human-review volume to staffing capacity;
- a calibration curve testing whether confidence scores deserve trust;
- an unsupervised TF-IDF → SVD → k-means map of natural issue themes;
- bootstrap and Wilson intervals communicating statistical uncertainty;
- RAG similarity and threshold plots comparing relevant and out-of-domain questions;
- a live case explorer showing classification signals, cited guidance, and abstention;
- a student journey explaining what I learned and what I would build next.

The map uses clearly labeled simulated locations and capacity. The model diagnostics, uncertainty
estimates, and RAG results are calculated from the real held-out benchmark predictions.

## My learning journey

| Stage | What changed |
|---|---|
| December 2025 | Built my first data science/AI project at a Break Through Tech workshop |
| First iteration | Learned preprocessing, TF-IDF, classification, and basic evaluation |
| Rebuild | Added issue-family holdout testing, explainability, retrieval, citations, and an API |
| Outcome analysis | Added calibration, uncertainty intervals, clustering, and error analysis |
| Practical layer | Connected model abstention to a simulated staffing and location decision |
| Next step | Evaluate real consented data, test fairness and drift, and run a shadow deployment |

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
