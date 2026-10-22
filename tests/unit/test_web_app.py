import json

from fastapi.testclient import TestClient

from field_incident_agent.database import Database
from field_incident_agent.llm_client import ModelClientError
from field_incident_agent.runtime import build_runtime
from field_incident_agent.web_app import create_app


def extraction(**overrides):
    result = {
        "site_id": "SITE-DEMO",
        "asset_id": "DEMO-PUMP-001",
        "asset_lookup_clue": None,
        "description": "The demonstration pump is vibrating and output is reduced.",
        "symptoms": ["vibration", "reduced output"],
        "safety_indicators": ["equipment_degraded"],
        "safety_concern_uncertain": False,
        "time_noticed": None,
        "operational_impact": "Reduced pumping output",
    }
    result.update(overrides)
    return result


class FakeClient:
    model = "fake-test-model"

    def __init__(self, response):
        self.response = response

    def extract_incident(self, text):
        if isinstance(self.response, Exception):
            raise self.response
        return json.dumps(self.response)


def make_client(tmp_path, response):
    database_path = tmp_path / "web-demo.sqlite3"
    runtime = build_runtime(database_path, client=FakeClient(response))
    test_client = TestClient(create_app(runtime=runtime))
    return test_client, runtime


def table_count(runtime, table):
    with runtime.database.connect() as connection:
        return connection.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]


def test_home_serves_voice_and_text_interface(tmp_path):
    client, _ = make_client(tmp_path, extraction())
    response = client.get("/")

    assert response.status_code == 200
    assert "Start voice input" in response.text
    assert "Send message" in response.text
    assert "Confirm and create ticket" in response.text


def test_ticket_requires_confirmation_and_duplicate_confirm_is_idempotent(
    tmp_path,
):
    client, runtime = make_client(tmp_path, extraction())

    intake = client.post(
        "/api/intake",
        json={
            "text": (
                "At SITE-DEMO, DEMO-PUMP-001 is vibrating and output is reduced."
            )
        },
    )

    assert intake.status_code == 200
    proposal = intake.json()
    assert proposal["status"] == "confirmation_required"
    assert proposal["proposal_id"]
    assert table_count(runtime, "incidents") == 0
    assert table_count(runtime, "confirmation_events") == 0

    url = f"/api/proposals/{proposal['proposal_id']}/confirm"
    confirmed = client.post(url, json={"confirmed": True})

    assert confirmed.status_code == 200
    first_result = confirmed.json()
    assert first_result["status"] == "ticket_created"
    assert table_count(runtime, "incidents") == 1
    assert table_count(runtime, "confirmation_events") == 1

    replay = client.post(url, json={"confirmed": True})

    assert replay.status_code == 200
    assert replay.json()["data"]["incident_id"] == (
        first_result["data"]["incident_id"]
    )
    assert replay.json()["data"]["replayed"] is True
    assert table_count(runtime, "incidents") == 1


def test_critical_safety_report_never_offers_normal_confirmation(tmp_path):
    client, runtime = make_client(
        tmp_path,
        extraction(
            site_id=None,
            asset_id=None,
            asset_lookup_clue=None,
            safety_indicators=["fire_or_smoke"],
        ),
    )

    response = client.post(
        "/api/intake",
        json={"text": "Smoke and visible flames are coming from the panel."},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "safety_handoff"
    assert "proposal_id" not in response.json()
    assert table_count(runtime, "incidents") == 0
    assert table_count(runtime, "confirmation_events") == 0


def test_model_failure_returns_unavailable_without_creating_ticket(tmp_path):
    client, runtime = make_client(
        tmp_path,
        ModelClientError("simulated provider outage"),
    )

    response = client.post(
        "/api/intake",
        json={"text": "Synthetic pump report."},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "model_unavailable"
    assert table_count(runtime, "incidents") == 0


def test_unknown_proposal_cannot_create_ticket(tmp_path):
    client, runtime = make_client(tmp_path, extraction())

    response = client.post(
        "/api/proposals/not-a-valid-proposal/confirm",
        json={"confirmed": True},
    )

    assert response.status_code == 404
    assert table_count(runtime, "incidents") == 0


def test_extra_request_fields_are_rejected(tmp_path):
    client, runtime = make_client(tmp_path, extraction())

    response = client.post(
        "/api/intake",
        json={
            "text": "Synthetic pump report.",
            "severity": "low",
            "confirmed": True,
        },
    )

    assert response.status_code == 422
    assert table_count(runtime, "incidents") == 0