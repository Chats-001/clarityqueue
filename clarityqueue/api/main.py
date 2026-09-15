"""FastAPI entry point for triage, retrieval, answers, and metrics."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from clarityqueue.api.schemas import (
    AnswerResponse,
    EvidenceResponse,
    SearchRequest,
    TicketRequest,
    TriageResponse,
)
from clarityqueue.api.service import ClarityService
from clarityqueue.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.service = ClarityService.from_artifacts(
            settings.artifact_dir, settings.event_db, settings.answer_threshold
        )
    except (FileNotFoundError, ValueError):
        app.state.service = None
    yield


app = FastAPI(
    title="ClarityQueue API",
    version="0.1.0",
    description="Measurable ticket triage and evidence-grounded operational guidance.",
    lifespan=lifespan,
)


def get_service(request: Request) -> ClarityService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(
            status_code=503, detail="Artifacts unavailable; run the training workflow"
        )
    return service


@app.get("/health")
def health(request: Request) -> dict:
    service = getattr(request.app.state, "service", None)
    return {
        "status": "ok" if service else "degraded",
        "model_version": service.version if service else None,
    }


@app.post("/v1/triage", response_model=TriageResponse)
def triage(ticket: TicketRequest, service: ClarityService = Depends(get_service)) -> dict:
    return service.triage(ticket)


@app.post("/v1/retrieve", response_model=list[EvidenceResponse])
def retrieve(request: SearchRequest, service: ClarityService = Depends(get_service)) -> list[dict]:
    return service.retrieve(request.query, request.limit)


@app.post("/v1/answer", response_model=AnswerResponse)
def answer(ticket: TicketRequest, service: ClarityService = Depends(get_service)) -> dict:
    return service.answer(ticket)


@app.get("/v1/metrics")
def metrics(service: ClarityService = Depends(get_service)) -> dict:
    return {**service.metrics, "model_version": service.version}
