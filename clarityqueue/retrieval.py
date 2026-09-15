"""Measured evidence retrieval and an explicit low-confidence abstention policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class EvidenceHit:
    article_id: str
    route: str
    title: str
    summary: str
    steps: list[str]
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


class EvidenceIndex:
    """A transparent sparse retriever over a curated operational knowledge base."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.articles = pd.DataFrame()
        self.matrix = None

    def fit(self, articles: pd.DataFrame) -> EvidenceIndex:
        required = {"article_id", "route", "title", "summary", "steps"}
        missing = required - set(articles.columns)
        if missing:
            raise ValueError(f"Knowledge base is missing: {', '.join(sorted(missing))}")
        if articles["article_id"].duplicated().any():
            raise ValueError("Knowledge article IDs must be unique")
        self.articles = articles.reset_index(drop=True).copy()
        documents = (
            self.articles["title"] + " " + self.articles["summary"] + " " + self.articles["steps"]
        )
        self.matrix = self.vectorizer.fit_transform(documents)
        return self

    def search(self, query: str, limit: int = 3, route: str | None = None) -> list[EvidenceHit]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        if limit < 1:
            raise ValueError("Limit must be positive")
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        eligible = np.arange(len(self.articles))
        if route is not None:
            eligible = eligible[self.articles.iloc[eligible]["route"].to_numpy() == route]
        ranked = eligible[np.argsort(scores[eligible])[::-1]][:limit]
        hits = []
        for index in ranked:
            row = self.articles.iloc[index]
            hits.append(
                EvidenceHit(
                    article_id=str(row.article_id),
                    route=str(row.route),
                    title=str(row.title),
                    summary=str(row.summary),
                    steps=[part.strip() for part in str(row.steps).split("|")],
                    score=float(scores[index]),
                )
            )
        return hits

    def answer(self, query: str, threshold: float = 0.16, route: str | None = None) -> dict:
        hits = self.search(query, limit=3, route=route)
        top = hits[0]
        if top.score < threshold:
            return {
                "answered": False,
                "answer": (
                    "I could not find strong enough evidence. Route this case to a human reviewer."
                ),
                "confidence": top.score,
                "citations": [],
                "alternatives": [hit.to_dict() for hit in hits],
            }
        guidance = " ".join(f"{number}. {step}" for number, step in enumerate(top.steps, 1))
        return {
            "answered": True,
            "answer": f"Recommended runbook: {top.title}. {guidance}",
            "confidence": top.score,
            "citations": [{"article_id": top.article_id, "title": top.title}],
            "alternatives": [hit.to_dict() for hit in hits[1:]],
        }
