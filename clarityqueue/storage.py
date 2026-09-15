"""Privacy-conscious SQLite event logging for API analytics."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path


def initialize(path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(target) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS inference_event (
                event_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                model_version TEXT NOT NULL,
                route TEXT,
                priority TEXT,
                confidence REAL,
                answered INTEGER,
                latency_ms REAL NOT NULL,
                metadata_json TEXT NOT NULL
            )"""
        )


def log_event(
    path: str | Path, event_type: str, model_version: str, latency_ms: float, **values
) -> str:
    event_id = str(uuid.uuid4())
    known = {key: values.pop(key, None) for key in ["route", "priority", "confidence", "answered"]}
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO inference_event VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                datetime.now(UTC).isoformat(),
                event_type,
                model_version,
                known["route"],
                known["priority"],
                known["confidence"],
                known["answered"],
                latency_ms,
                json.dumps(values, sort_keys=True),
            ),
        )
    return event_id
