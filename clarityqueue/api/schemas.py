"""Validated API contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from clarityqueue.config import settings


class TicketRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=3, max_length=200)
    description: str = Field(min_length=5, max_length=settings.max_text_length)

    @field_validator("title", "description")
    @classmethod
    def meaningful_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text must not be blank")
        return cleaned


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=3, max_length=settings.max_text_length)
    limit: int = Field(default=3, ge=1, le=settings.max_results)


class TriageResponse(BaseModel):
    route: str
    route_confidence: float
    priority: str
    priority_confidence: float
    signals: list[str]
    model_version: str


class EvidenceResponse(BaseModel):
    article_id: str
    route: str
    title: str
    summary: str
    steps: list[str]
    score: float


class AnswerResponse(BaseModel):
    answered: bool
    answer: str
    confidence: float
    citations: list[dict]
    alternatives: list[dict]
    predicted_route: str
    model_version: str
