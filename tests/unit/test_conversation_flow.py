import json

from fastapi.testclient import TestClient

from field_incident_agent.database import Database
from field_incident_agent.runtime import build_runtime
from field_incident_agent.web_app import create_app


def extraction(**overrides):
    result = {
        "site_id": "SITE-DEMO",
        "asset_id": "DEMO-PUMP-001",
        "asset_lookup_clue": None,
        "description": (
            "The main demonstration pump is vibrating and output is reduced."
        ),
        "symptoms": ["vibration", "reduced output"],
        "safety_indicators": ["equipment_degraded"],
        "safety_concern_uncertain": False,
        "time_noticed": None,
        "operational_impact": "Reduced pumping output",
    }
    result.update(overrides)
    return result


class FakeClient:
    model = "fake-conversation-test"

    def __init__(self, response):
        self.response = response
        self.calls = []

    def extract_incident(self, text):
        self.calls.append(text)
        value = self.response(text) if callable(self.response) else self.response
        return json.dumps(value)


def make_client(tmp_path, response):
    database_path = tmp_path / "conversation-demo.db"
    fake = FakeClient(response)
    runtime = build_runtime(database_path, client=fake)
    client = TestClient(create_app(runtime=runtime))
    return client, runtime, fake


def table_count(runtime, table):
    with runtime.database.connect() as connection:
        return connection.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]


def test_first_turn_returns_a_natural_proposal_not_debug_json(tmp_path):
    client, runtime, _ = make_client(tmp_path, extraction())

    response = client.post(
        "/api/turn",
        json={
            "text": (
                "At SITE-DEMO, DEMO-PUMP-001 is vibrating and output is reduced."
            )
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "confirmation_required"
    assert "Here's what I understood" in body["agent_message"]
    assert "medium severity" in body["agent_message"]
    assert "create the ticket" in body["agent_message"]
    assert "{" not in body["agent_message"]
    assert body["proposal"]["asset_id"] == "DEMO-PUMP-001"
    assert body["proposal_id"]
    assert table_count(runtime, "incidents") == 0


def test_saying_yes_confirms_the_current_proposal(tmp_path):
    client, runtime, _ = make_client(tmp_path, extraction())

    first = client.post(
        "/api/turn",
        json={"text": "The main demonstration pump is vibrating."},
    ).json()

    second = client.post(
        "/api/turn",
        json={
            "conversation_id": first["conversation_id"],
            "text": "Yes, create the ticket.",
        },
    )

    assert second.status_code == 200
    body = second.json()
    assert body["status"] == "ticket_created"
    assert "saved in the local demo database" in body["agent_message"]
    assert table_count(runtime, "incidents") == 1
    assert table_count(runtime, "confirmation_events") == 1


def test_correction_replaces_asset_and_invalidates_old_proposal(tmp_path):
    def respond(text):
        if "CALLER CORRECTION" in text:
            return extraction(
                asset_id="DEMO-PUMP-002",
                description=(
                    "The backup demonstration pump is vibrating and output is reduced."
                ),
            )
        return extraction()

    client, runtime, fake = make_client(tmp_path, respond)

    first = client.post(
        "/api/turn",
        json={"text": "The main pump is vibrating and output is reduced."},
    ).json()
    old_proposal = first["proposal_id"]

    corrected = client.post(
        "/api/turn",
        json={
            "conversation_id": first["conversation_id"],
            "text": "I meant the backup pump, DEMO-PUMP-002.",
        },
    )

    assert corrected.status_code == 200
    body = corrected.json()
    assert body["status"] == "confirmation_required"
    assert body["revision"] == 1
    assert body["proposal"]["asset_id"] == "DEMO-PUMP-002"
    assert "updated understanding" in body["agent_message"]
    assert body["proposal_id"] != old_proposal
    assert "CALLER CORRECTION" in fake.calls[-1]
    assert table_count(runtime, "incidents") == 0

    stale = client.post(
        f"/api/proposals/{old_proposal}/confirm",
        json={"confirmed": True},
    )
    assert stale.status_code == 404
    assert table_count(runtime, "incidents") == 0


def test_declining_proposal_never_creates_ticket(tmp_path):
    client, runtime, _ = make_client(tmp_path, extraction())

    first = client.post(
        "/api/turn",
        json={"text": "The main pump output is reduced."},
    ).json()

    declined = client.post(
        "/api/turn",
        json={
            "conversation_id": first["conversation_id"],
            "text": "No, that's wrong.",
        },
    )

    assert declined.status_code == 200
    body = declined.json()
    assert body["status"] == "confirmation_required"
    assert body["proposal"]["asset_id"] == "DEMO-PUMP-001"
    assert table_count(runtime, "incidents") == 0

    explicitly_declined = client.post(
        "/api/turn",
        json={
            "conversation_id": first["conversation_id"],
            "text": "No",
        },
    )

    assert explicitly_declined.status_code == 200
    assert explicitly_declined.json()["status"] == "awaiting_correction"
    assert table_count(runtime, "incidents") == 0
    assert table_count(runtime, "confirmation_events") == 0


def test_critical_report_stops_normal_ticket_flow(tmp_path):
    client, runtime, _ = make_client(
        tmp_path,
        extraction(
            site_id=None,
            asset_id=None,
            asset_lookup_clue=None,
            safety_indicators=["fire_or_smoke"],
        ),
    )

    response = client.post(
        "/api/turn",
        json={"text": "Smoke and visible flames are present."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "safety_handoff"
    assert "no normal incident ticket was created" in body["agent_message"]
    assert body["proposal_id"] is None
    assert table_count(runtime, "incidents") == 0
def test_no_after_safety_handoff_preserves_context(tmp_path):
    client, runtime, fake = make_client(
        tmp_path,
        extraction(
            site_id=None,
            asset_id=None,
            asset_lookup_clue=None,
            safety_indicators=["fire_or_smoke"],
        ),
    )

    first = client.post(
        "/api/turn",
        json={"text": "Smoke and visible flames are present."},
    ).json()

    assert first["status"] == "safety_handoff"

    followup = client.post(
        "/api/turn",
        json={
            "conversation_id": first["conversation_id"],
            "text": "No",
        },
    )

    assert followup.status_code == 200
    body = followup.json()
    assert body["conversation_id"] == first["conversation_id"]
    assert body["status"] == "safety_handoff"
    assert len(fake.calls) == 1
    assert "what" in body["agent_message"].lower()
    assert table_count(runtime, "incidents") == 0


def test_yes_after_safety_handoff_cannot_create_normal_ticket(tmp_path):
    client, runtime, fake = make_client(
        tmp_path,
        extraction(
            site_id=None,
            asset_id=None,
            asset_lookup_clue=None,
            safety_indicators=["fire_or_smoke"],
        ),
    )

    first = client.post(
        "/api/turn",
        json={"text": "Smoke and visible flames are present."},
    ).json()

    followup = client.post(
        "/api/turn",
        json={
            "conversation_id": first["conversation_id"],
            "text": "Yes, create the ticket.",
        },
    )

    assert followup.status_code == 200
    assert followup.json()["conversation_id"] == first["conversation_id"]
    assert followup.json()["status"] == "safety_handoff"
    assert len(fake.calls) == 1
    assert table_count(runtime, "incidents") == 0
    assert table_count(runtime, "confirmation_events") == 0


def test_no_after_ticket_creation_does_not_start_another_incident(tmp_path):
    client, runtime, fake = make_client(tmp_path, extraction())

    first = client.post(
        "/api/turn",
        json={"text": "The main demonstration pump is vibrating."},
    ).json()

    confirmed = client.post(
        "/api/turn",
        json={
            "conversation_id": first["conversation_id"],
            "text": "Yes, create the ticket.",
        },
    ).json()

    assert confirmed["status"] == "ticket_created"
    assert table_count(runtime, "incidents") == 1

    followup = client.post(
        "/api/turn",
        json={
            "conversation_id": first["conversation_id"],
            "text": "No",
        },
    )

    assert followup.status_code == 200
    body = followup.json()
    assert body["conversation_id"] == first["conversation_id"]
    assert body["status"] == "ticket_created"
    assert "already" in body["agent_message"].lower()
    assert len(fake.calls) == 1
    assert table_count(runtime, "incidents") == 1
