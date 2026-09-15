"""Interpretable text classifiers for routing and priority triage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def combine_text(title: str, description: str) -> str:
    return f"{title.strip()} {description.strip()}".strip()


def _pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            (
                "classifier",
                LogisticRegression(max_iter=1_500, class_weight="balanced", random_state=17),
            ),
        ]
    )


@dataclass
class TriagePrediction:
    route: str
    route_confidence: float
    priority: str
    priority_confidence: float
    signals: list[str]


class TriageModel:
    """Two compact, independently measurable classifiers with token-level signals."""

    def __init__(self) -> None:
        self.route_model = _pipeline()
        self.priority_model = _pipeline()

    def fit(self, texts, routes, priorities) -> TriageModel:
        self.route_model.fit(texts, routes)
        self.priority_model.fit(texts, priorities)
        return self

    def predict_many(self, texts) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        route_probabilities = self.route_model.predict_proba(texts)
        priority_probabilities = self.priority_model.predict_proba(texts)
        route_indices = route_probabilities.argmax(axis=1)
        priority_indices = priority_probabilities.argmax(axis=1)
        return (
            self.route_model.classes_[route_indices],
            route_probabilities.max(axis=1),
            self.priority_model.classes_[priority_indices],
            priority_probabilities.max(axis=1),
        )

    def predict(self, title: str, description: str) -> TriagePrediction:
        text = combine_text(title, description)
        routes, route_scores, priorities, priority_scores = self.predict_many([text])
        return TriagePrediction(
            route=str(routes[0]),
            route_confidence=float(route_scores[0]),
            priority=str(priorities[0]),
            priority_confidence=float(priority_scores[0]),
            signals=self.route_signals(text, str(routes[0])),
        )

    def route_signals(self, text: str, predicted_route: str, limit: int = 5) -> list[str]:
        vectorizer = self.route_model.named_steps["tfidf"]
        classifier = self.route_model.named_steps["classifier"]
        row = vectorizer.transform([text])
        class_index = list(classifier.classes_).index(predicted_route)
        contributions = row.multiply(classifier.coef_[class_index]).toarray()[0]
        terms = vectorizer.get_feature_names_out()
        positive = np.flatnonzero(contributions > 0)
        ranked = positive[np.argsort(contributions[positive])[::-1]][:limit]
        return [str(terms[index]) for index in ranked]
