from fastapi.testclient import TestClient

from clarityqueue.api.main import app, get_service
from clarityqueue.api.service import ClarityService
from clarityqueue.storage import initialize


def client_for(tmp_path, triage_model, evidence_index):
    database = tmp_path / "events.sqlite3"
    initialize(database)
    service = ClarityService(
        triage_model,
        evidence_index,
        {"model_version": "test-v1"},
        {"route_model": {"accuracy": 0.9}},
        database,
        0.1,
    )
    app.dependency_overrides[get_service] = lambda: service
    app.state.service = service
    return TestClient(app)


def test_health(tmp_path, triage_model, evidence_index):
    client = client_for(tmp_path, triage_model, evidence_index)
    assert client.get("/health").json() == {"status": "ok", "model_version": "test-v1"}


def test_triage_endpoint(tmp_path, triage_model, evidence_index):
    client = client_for(tmp_path, triage_model, evidence_index)
    response = client.post(
        "/v1/triage",
        json={
            "title": "API token expired",
            "description": "Requests return 401 authentication errors",
        },
    )
    assert response.status_code == 200
    assert response.json()["route"] == "integration"
    assert response.json()["model_version"] == "test-v1"


def test_retrieve_endpoint(tmp_path, triage_model, evidence_index):
    client = client_for(tmp_path, triage_model, evidence_index)
    response = client.post("/v1/retrieve", json={"query": "invalid webhook HMAC", "limit": 2})
    assert response.status_code == 200
    assert response.json()[0]["article_id"] == "KB-INT-01"


def test_answer_endpoint_has_source(tmp_path, triage_model, evidence_index):
    client = client_for(tmp_path, triage_model, evidence_index)
    response = client.post(
        "/v1/answer",
        json={"title": "Lost MFA device", "description": "I cannot enter my authenticator code"},
    )
    assert response.status_code == 200
    assert response.json()["citations"]


def test_blank_title_is_rejected(tmp_path, triage_model, evidence_index):
    client = client_for(tmp_path, triage_model, evidence_index)
    response = client.post("/v1/triage", json={"title": "   ", "description": "valid body"})
    assert response.status_code == 422


def test_missing_description_is_rejected(tmp_path, triage_model, evidence_index):
    client = client_for(tmp_path, triage_model, evidence_index)
    assert client.post("/v1/triage", json={"title": "valid title"}).status_code == 422


def test_extra_field_is_rejected(tmp_path, triage_model, evidence_index):
    client = client_for(tmp_path, triage_model, evidence_index)
    payload = {"title": "valid title", "description": "valid description", "customer": "secret"}
    assert client.post("/v1/triage", json=payload).status_code == 422


def test_retrieve_limit_is_bounded(tmp_path, triage_model, evidence_index):
    client = client_for(tmp_path, triage_model, evidence_index)
    assert client.post("/v1/retrieve", json={"query": "API errors", "limit": 99}).status_code == 422


def test_blank_retrieval_query_is_rejected(tmp_path, triage_model, evidence_index):
    client = client_for(tmp_path, triage_model, evidence_index)
    response = client.post("/v1/retrieve", json={"query": "   "})
    assert response.status_code == 422


def test_metrics_endpoint(tmp_path, triage_model, evidence_index):
    client = client_for(tmp_path, triage_model, evidence_index)
    response = client.get("/v1/metrics")
    assert response.json()["route_model"]["accuracy"] == 0.9
