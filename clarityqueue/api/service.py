"""Framework-independent application service."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from clarityqueue.api.schemas import TicketRequest
from clarityqueue.model import combine_text
from clarityqueue.persistence import load_bundle
from clarityqueue.storage import initialize, log_event


@dataclass
class ClarityService:
    model: object
    index: object
    metadata: dict
    metrics: dict
    event_db: Path
    answer_threshold: float

    @classmethod
    def from_artifacts(
        cls, artifact_dir: Path, event_db: Path, answer_threshold: float
    ) -> ClarityService:
        model, index, metadata, metrics = load_bundle(artifact_dir)
        initialize(event_db)
        return cls(model, index, metadata, metrics, event_db, answer_threshold)

    @property
    def version(self) -> str:
        return str(self.metadata["model_version"])

    def triage(self, ticket: TicketRequest) -> dict:
        start = time.perf_counter()
        result = self.model.predict(ticket.title, ticket.description)
        latency = (time.perf_counter() - start) * 1_000
        payload = {
            "route": result.route,
            "route_confidence": result.route_confidence,
            "priority": result.priority,
            "priority_confidence": result.priority_confidence,
            "signals": result.signals,
            "model_version": self.version,
        }
        log_event(
            self.event_db,
            "triage",
            self.version,
            latency,
            route=result.route,
            priority=result.priority,
            confidence=result.route_confidence,
            signal_count=len(result.signals),
        )
        return payload

    def retrieve(self, query: str, limit: int) -> list[dict]:
        start = time.perf_counter()
        hits = self.index.search(query, limit=limit)
        latency = (time.perf_counter() - start) * 1_000
        log_event(
            self.event_db,
            "retrieve",
            self.version,
            latency,
            confidence=hits[0].score,
            result_count=len(hits),
        )
        return [hit.to_dict() for hit in hits]

    def answer(self, ticket: TicketRequest) -> dict:
        start = time.perf_counter()
        triage = self.model.predict(ticket.title, ticket.description)
        query = combine_text(ticket.title, ticket.description)
        result = self.index.answer(query, self.answer_threshold, route=triage.route)
        latency = (time.perf_counter() - start) * 1_000
        result.update({"predicted_route": triage.route, "model_version": self.version})
        log_event(
            self.event_db,
            "answer",
            self.version,
            latency,
            route=triage.route,
            priority=triage.priority,
            confidence=result["confidence"],
            answered=int(result["answered"]),
            citation_count=len(result["citations"]),
        )
        return result
