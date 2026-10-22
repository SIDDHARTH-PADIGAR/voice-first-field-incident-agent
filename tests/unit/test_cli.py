import json

import pytest

from field_incident_agent.cli import main
from field_incident_agent.database import Database


class FakeExtractionClient:
    model = "fake-test-model"

    def __init__(self, response):
        self.response = response

    def extract_incident(self, text):
        return self.response


def extraction(**overrides):
    result = {
        "site_id": "SITE-DEMO",
        "asset_id": "DEMO-PUMP-001",
        "asset_lookup_clue": None,
        "description": (
            "The main demonstration pump is vibrating and its output is reduced."
        ),
        "symptoms": ["vibration", "reduced output"],
        "safety_indicators": ["equipment_degraded"],
        "safety_concern_uncertain": False,
        "time_noticed": None,
        "operational_impact": "Reduced pumping output",
    }
    result.update(overrides)
    return result


def test_demo_creates_ticket_only_after_explicit_yes(
    tmp_path, monkeypatch, capsys
):
    database_path = tmp_path / "demo.db"
    answers = iter([
        (
            "At SITE-DEMO, DEMO-PUMP-001 is vibrating and output is reduced. "
            "No smoke, sparks, leak, injury, or immediate danger."
        ),
        "YES",
    ])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    result = main(
        ["--database", str(database_path)],
        client=FakeExtractionClient(extraction()),
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "Status: confirmation_required" in output
    assert "Final status: ticket_created" in output

    database = Database(database_path)
    with database.connect() as connection:
        incidents = connection.execute(
            "SELECT COUNT(*) FROM incidents"
        ).fetchone()[0]
        confirmations = connection.execute(
            "SELECT COUNT(*) FROM confirmation_events"
        ).fetchone()[0]

    assert incidents == 1
    assert confirmations == 1


def test_critical_safety_report_goes_to_handoff_without_ticket(
    tmp_path, monkeypatch, capsys
):
    database_path = tmp_path / "critical-demo.db"
    answers = iter([
        "Smoke and visible flames are coming from the electrical panel."
    ])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    result = main(
        ["--database", str(database_path)],
        client=FakeExtractionClient(
            extraction(
                asset_id=None,
                asset_lookup_clue=None,
                safety_indicators=["fire_or_smoke"],
            )
        ),
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "Status: safety_handoff" in output
    assert "Final status: ticket_created" not in output

    database = Database(database_path)
    with database.connect() as connection:
        incidents = connection.execute(
            "SELECT COUNT(*) FROM incidents"
        ).fetchone()[0]

    assert incidents == 0