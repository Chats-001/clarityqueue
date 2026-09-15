"""Shared application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROUTES = ("access", "billing", "data_sync", "integration", "performance")
PRIORITIES = ("low", "medium", "high", "critical")
RANDOM_SEED = 17


@dataclass(frozen=True)
class Settings:
    artifact_dir: Path = Path(os.getenv("CLARITYQUEUE_ARTIFACT_DIR", "artifacts"))
    event_db: Path = Path(os.getenv("CLARITYQUEUE_EVENT_DB", "artifacts/events.sqlite3"))
    answer_threshold: float = float(os.getenv("CLARITYQUEUE_ANSWER_THRESHOLD", "0.16"))
    max_text_length: int = int(os.getenv("CLARITYQUEUE_MAX_TEXT_LENGTH", "4000"))
    max_results: int = int(os.getenv("CLARITYQUEUE_MAX_RESULTS", "5"))


settings = Settings()
