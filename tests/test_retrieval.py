import pandas as pd
import pytest

from clarityqueue.retrieval import EvidenceIndex


def test_signature_query_returns_signature_article(evidence_index):
    hit = evidence_index.search("invalid HMAC webhook signature", limit=1)[0]
    assert hit.article_id == "KB-INT-01"


def test_search_limit_is_respected(evidence_index):
    assert len(evidence_index.search("API error", limit=2)) == 2


def test_scores_are_sorted(evidence_index):
    scores = [hit.score for hit in evidence_index.search("slow API latency", limit=5)]
    assert scores == sorted(scores, reverse=True)


def test_route_filter_is_respected(evidence_index):
    hits = evidence_index.search("error", limit=3, route="billing")
    assert all(hit.route == "billing" for hit in hits)


def test_empty_query_is_rejected(evidence_index):
    with pytest.raises(ValueError, match="empty"):
        evidence_index.search("   ")


def test_invalid_limit_is_rejected(evidence_index):
    with pytest.raises(ValueError, match="positive"):
        evidence_index.search("query", limit=0)


def test_duplicate_article_ids_are_rejected(articles):
    duplicate = pd.concat([articles, articles.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        EvidenceIndex().fit(duplicate)


def test_grounded_answer_has_citation(evidence_index):
    answer = evidence_index.answer("I lost my MFA authenticator device", threshold=0.05)
    assert answer["answered"] is True
    assert answer["citations"][0]["article_id"] == "KB-ACCESS-02"


def test_low_evidence_answer_abstains(evidence_index):
    answer = evidence_index.answer("purple weather banana orchestra", threshold=0.9)
    assert answer["answered"] is False
    assert answer["citations"] == []


def test_steps_are_structured(evidence_index):
    hit = evidence_index.search("invoice tax address", limit=1)[0]
    assert len(hit.steps) >= 3
