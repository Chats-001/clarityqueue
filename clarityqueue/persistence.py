"""Save and load a versioned ML plus retrieval bundle."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib


def write_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def content_hash(*paths: str | Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def save_bundle(model, index, metrics: dict, artifact_dir: str | Path, source_hash: str) -> dict:
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    version = f"cq-{datetime.now(UTC).strftime('%Y%m%d')}-{source_hash[:8]}"
    joblib.dump({"model": model, "index": index}, directory / "bundle.joblib")
    metadata = {
        "model_version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "source_sha256": source_hash,
        "approach": "TF-IDF logistic regression plus evidence retrieval",
    }
    write_json(directory / "metadata.json", metadata)
    write_json(directory / "metrics.json", metrics)
    return metadata


def load_bundle(artifact_dir: str | Path) -> tuple[object, object, dict, dict]:
    directory = Path(artifact_dir)
    bundle = joblib.load(directory / "bundle.joblib")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    return bundle["model"], bundle["index"], metadata, metrics
