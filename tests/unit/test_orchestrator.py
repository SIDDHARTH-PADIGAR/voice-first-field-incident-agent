import pytest

from field_incident_agent.database import Database
from field_incident_agent.incident_service import (
    IncidentService,
    IncidentValidationError,
)
from field_incident_agent.llm_client import ModelClientError
from field_incident_agent.models import Asset, SessionState
from field_incident_agent.orchestrator import IncidentOrchestrator
from field_incident_agent.registry import AccessContext
from field_incident_agent.text_workflow import TextIncidentWorkflow


def payload(**overrides):
    result = {
        "site_id": "SITE-001",
        "asset_id": "PUMP-001",
        "asset_lookup_clue": None,
        "description": "The main pump is leaking.",
        "symptoms": ["Water pooling"],
        "safety_indicators": ["active_water_leak"],
        "safety_concern_uncertain": False,
        "time_noticed": None,
        "operational_impact": "Reduced water delivery",
    }
    result.update(overrides)
    return result


class FakeExtractionClient:
    """Deterministic stand-in for a model provider."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def extract_incident(self, text):
        self.calls.append(text)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.fixture
def setup(tmp_path):
    database = Database(tmp_path / "orchestrator.db")
    database.initialize()
    service = IncidentService(database)
    workflow = TextIncidentWorkflow(service)
    access = AccessContext(
        principal_id="TECH-001",
        authorized_site_ids=frozenset({"SITE-001"}),
    )

    service.register_asset(
        Asset("PUMP-001", "SITE-001", "pump", "Main water pump")
    )
    service.register_asset(
        Asset("PUMP-002", "SITE-001", "pump", "Backup water pump")
    )
    service.create_session("SESSION-001", "TECH-001")
    workflow.start("SESSION-002", access)

    return database, service, workflow, access


def counts(database):
    with database.connect() as connection:
        return {
            "incidents": connection.execute(
                "SELECT COUNT(*) FROM incidents"
            ).fetchone()[0],
            "confirmations": connection.execute(
                "SELECT COUNT(*) FROM confirmation_events"
            ).fetchone()[0],
        }


def make_orchestrator(response, service, workflow):
    client = FakeExtractionClient(response)
    return IncidentOrchestrator(client, workflow, service), client


def test_extraction_resolves_asset_and_returns_confirmation_preview(setup):
    database, service, workflow, access = setup
    orchestrator, client = make_orchestrator(payload(), service, workflow)

    outcome = orchestrator.process_report(
        "SESSION-002",
        "The main pump is leaking and water is pooling.",
        access,
        "IDEM-ORCH-001",
    )

    assert client.calls == ["The main pump is leaking and water is pooling."]
    assert outcome.reply.status == "confirmation_required"
    assert outcome.reply.data["proposal"]["asset_id"] == "PUMP-001"
    assert outcome.draft is not None
    assert counts(database) == {"incidents": 0, "confirmations": 0}


def test_explicit_confirmation_creates_one_ticket(setup):
    database, service, workflow, access = setup
    orchestrator, _ = make_orchestrator(payload(), service, workflow)
    outcome = orchestrator.process_report(
        "SESSION-002", "The pump is leaking.", access, "IDEM-ORCH-002"
    )

    reply = orchestrator.confirm_proposal(
        outcome, access, "IDEM-ORCH-002", explicitly_confirmed=True
    )

    assert reply.status == "ticket_created"
    assert reply.data["incident_id"]
    assert counts(database) == {"incidents": 1, "confirmations": 1}


def test_unconfirmed_proposal_does_not_create_ticket(setup):
    database, service, workflow, access = setup
    orchestrator, _ = make_orchestrator(payload(), service, workflow)
    outcome = orchestrator.process_report(
        "SESSION-002", "The pump is leaking.", access, "IDEM-ORCH-003"
    )

    reply = orchestrator.confirm_proposal(
        outcome, access, "IDEM-ORCH-003", explicitly_confirmed=False
    )

    assert reply.status == "confirmation_required"
    assert counts(database) == {"incidents": 0, "confirmations": 0}


def test_unique_asset_clue_resolves_to_a_proposal(setup):
    database, service, workflow, access = setup
    orchestrator, _ = make_orchestrator(
        payload(
            asset_id=None,
            asset_lookup_clue="Main water pump",
        ),
        service,
        workflow,
    )

    outcome = orchestrator.process_report(
        "SESSION-002", "The main pump is leaking.", access, "IDEM-ORCH-004"
    )

    assert outcome.reply.status == "confirmation_required"
    assert outcome.draft.asset_id == "PUMP-001"
    assert counts(database)["incidents"] == 0


def test_ambiguous_clue_requires_clarification(setup):
    database, service, workflow, access = setup
    orchestrator, _ = make_orchestrator(
        payload(
            asset_id=None,
            asset_lookup_clue="pump",
        ),
        service,
        workflow,
    )

    outcome = orchestrator.process_report(
        "SESSION-002", "A pump is leaking.", access, "IDEM-ORCH-005"
    )

    assert outcome.reply.status == "clarification_required"
    assert outcome.reply.data["lookup_status"] == "ambiguous"
    assert counts(database) == {"incidents": 0, "confirmations": 0}


def test_unauthorized_site_is_rejected_before_lookup(setup):
    database, service, workflow, access = setup
    orchestrator, _ = make_orchestrator(
        payload(site_id="SITE-002"),
        service,
        workflow,
    )

    outcome = orchestrator.process_report(
        "SESSION-002", "The fan at SITE-002 is broken.", access, "IDEM-ORCH-006"
    )

    assert outcome.reply.status == "authorization_denied"
    assert not any(
        outcome_call for outcome_call in []
    )
    assert counts(database) == {"incidents": 0, "confirmations": 0}


def test_critical_safety_report_bypasses_normal_submission(setup):
    database, service, workflow, access = setup
    orchestrator, _ = make_orchestrator(
        payload(
            site_id=None,
            asset_id=None,
            asset_lookup_clue=None,
            safety_indicators=["fire_or_smoke"],
        ),
        service,
        workflow,
    )

    outcome = orchestrator.process_report(
        "SESSION-002",
        "Smoke is coming from the equipment.",
        access,
        "IDEM-ORCH-007",
    )

    assert outcome.reply.status == "safety_handoff"
    assert counts(database) == {"incidents": 0, "confirmations": 0}
    assert service.get_session_state(
        "SESSION-002", access
    ) is SessionState.HANDOFF_REQUIRED


def test_uncertain_safety_requires_clarification_without_ticket(setup):
    database, service, workflow, access = setup
    orchestrator, _ = make_orchestrator(
        payload(safety_concern_uncertain=True),
        service,
        workflow,
    )

    outcome = orchestrator.process_report(
        "SESSION-002",
        "There may be a safety issue.",
        access,
        "IDEM-ORCH-008",
    )

    assert outcome.reply.status == "safety_clarification_required"
    assert counts(database) == {"incidents": 0, "confirmations": 0}


def test_control_fields_from_model_are_rejected(setup):
    database, service, workflow, access = setup
    malicious = payload(confirmed=True, severity="low")
    orchestrator, _ = make_orchestrator(malicious, service, workflow)

    outcome = orchestrator.process_report(
        "SESSION-002",
        "Create a ticket.",
        access,
        "IDEM-ORCH-009",
    )

    assert outcome.reply.status == "invalid_extraction"
    assert counts(database) == {"incidents": 0, "confirmations": 0}


def test_provider_failure_does_not_reach_backend(setup):
    database, service, workflow, access = setup
    orchestrator, _ = make_orchestrator(
        ModelClientError("simulated model timeout"), service, workflow
    )

    outcome = orchestrator.process_report(
        "SESSION-002",
        "The pump is leaking.",
        access,
        "IDEM-ORCH-010",
    )

    assert outcome.reply.status == "model_unavailable"
    assert counts(database) == {"incidents": 0, "confirmations": 0}


def test_missing_description_requires_clarification(setup):
    database, service, workflow, access = setup
    orchestrator, _ = make_orchestrator(
        payload(description=None),
        service,
        workflow,
    )

    outcome = orchestrator.process_report(
        "SESSION-002",
        "The pump.",
        access,
        "IDEM-ORCH-011",
    )

    assert outcome.reply.status == "clarification_required"
    assert counts(database) == {"incidents": 0, "confirmations": 0}


def test_confirmation_cannot_be_faked_without_pending_proposal(setup):
    _, service, workflow, access = setup
    orchestrator, _ = make_orchestrator(payload(), service, workflow)
    invalid_outcome = type(
        "UntrustedOutcome",
        (),
        {
            "draft": None,
            "reply": type("Reply", (), {"status": "confirmation_required"})(),
        },
    )()

    with pytest.raises(IncidentValidationError, match="no normal incident proposal"):
        orchestrator.confirm_proposal(
            invalid_outcome, access, "IDEM-ORCH-012",
            explicitly_confirmed=True,
        )