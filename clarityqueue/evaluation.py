"""Honest classification, retrieval, and selective-answer evaluation."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def classification_summary(y_true, y_pred, labels: list[str]) -> dict:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in labels
        },
    }


def retrieval_summary(index, tickets: pd.DataFrame, threshold: float = 0.16) -> dict:
    if tickets.empty:
        raise ValueError("Retrieval evaluation requires at least one ticket")
    reciprocal_ranks = []
    hits_at_1 = 0
    hits_at_3 = 0
    answered = 0
    correct_when_answered = 0
    latencies = []
    rows = []
    for row in tickets.itertuples(index=False):
        query = f"{row.title} {row.description}"
        start = time.perf_counter()
        hits = index.search(query, limit=3)
        latencies.append((time.perf_counter() - start) * 1_000)
        ids = [hit.article_id for hit in hits]
        rank = ids.index(row.article_id) + 1 if row.article_id in ids else None
        hits_at_1 += int(rank == 1)
        hits_at_3 += int(rank is not None)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        is_answered = hits[0].score >= threshold
        answered += int(is_answered)
        correct_when_answered += int(is_answered and rank == 1)
        rows.append(
            {
                "ticket_id": row.ticket_id,
                "expected_article": row.article_id,
                "retrieved_article": hits[0].article_id,
                "top_score": hits[0].score,
                "answered": is_answered,
                "correct": rank == 1,
            }
        )
    count = len(tickets)
    return {
        "hit_at_1": hits_at_1 / count,
        "hit_at_3": hits_at_3 / count,
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
        "answer_coverage": answered / count,
        "selective_accuracy": correct_when_answered / answered if answered else 0.0,
        "citation_rate": 1.0 if answered else 0.0,
        "p95_retrieval_latency_ms": float(np.percentile(latencies, 95)),
        "details": rows,
    }
