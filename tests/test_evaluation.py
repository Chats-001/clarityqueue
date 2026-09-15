import sqlite3

import pytest

from clarityqueue.evaluation import classification_summary, retrieval_summary
from clarityqueue.storage import initialize, log_event


def test_classification_summary_perfect_score():
    result = classification_summary(["a", "b"], ["a", "b"], ["a", "b"])
    assert result["accuracy"] == result["macro_f1"] == 1.0


def test_confusion_matrix_has_label_order():
    result = classification_summary(["a", "b"], ["b", "b"], ["a", "b"])
    assert result["confusion_matrix"] == [[0, 1], [0, 1]]


def test_retrieval_summary_is_bounded(evidence_index, tickets):
    result = retrieval_summary(evidence_index, tickets.head(10))
    for name in ["hit_at_1", "hit_at_3", "mean_reciprocal_rank", "answer_coverage"]:
        assert 0 <= result[name] <= 1


def test_retrieval_details_match_ticket_count(evidence_index, tickets):
    result = retrieval_summary(evidence_index, tickets.head(7))
    assert len(result["details"]) == 7


def test_retrieval_summary_rejects_empty_ticket_set(evidence_index, tickets):
    with pytest.raises(ValueError, match="at least one ticket"):
        retrieval_summary(evidence_index, tickets.iloc[0:0])


def test_storage_creates_event_table(tmp_path):
    path = tmp_path / "events.sqlite3"
    initialize(path)
    with sqlite3.connect(path) as connection:
        tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    assert ("inference_event",) in tables


def test_log_event_avoids_raw_ticket_text(tmp_path):
    path = tmp_path / "events.sqlite3"
    initialize(path)
    log_event(path, "triage", "v1", 2.5, route="billing", confidence=0.8)
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT event_type, route, confidence, latency_ms FROM inference_event"
        ).fetchone()
        columns = [item[1] for item in connection.execute("PRAGMA table_info(inference_event)")]
    assert row == ("triage", "billing", 0.8, 2.5)
    assert "description" not in columns
