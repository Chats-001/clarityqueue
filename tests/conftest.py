from __future__ import annotations

import pandas as pd
import pytest

from clarityqueue.model import TriageModel, combine_text
from clarityqueue.retrieval import EvidenceIndex
from scripts.generate_data import generate_tickets


@pytest.fixture(scope="session")
def articles() -> pd.DataFrame:
    return pd.read_csv("data/source/knowledge_base.csv")


@pytest.fixture(scope="session")
def tickets() -> pd.DataFrame:
    return generate_tickets(n_per_case=6)


@pytest.fixture(scope="session")
def evidence_index(articles) -> EvidenceIndex:
    return EvidenceIndex().fit(articles)


@pytest.fixture(scope="session")
def triage_model(tickets) -> TriageModel:
    texts = [combine_text(row.title, row.description) for row in tickets.itertuples(index=False)]
    return TriageModel().fit(texts, tickets["route"], tickets["priority"])
