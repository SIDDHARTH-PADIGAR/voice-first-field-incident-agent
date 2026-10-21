import pytest

from field_incident_agent.database import Database
from field_incident_agent.incident_service import (
    IncidentService,
    IncidentValidationError,
)
from field_incident_agent.models import Asset, IncidentDraft, SessionState
from field_incident_agent.registry import AccessContext
from field_incident_agent.text_workflow import TextIncidentWorkflow


@pytest.fixture
def setup(tmp_path):
    database = Database(tmp_path / "text-workflow.db")
    database.initialize()
    service = IncidentService(database)
    workflow = TextIncidentWorkflow(service)

    access = AccessContext(
        principal_id="TECH-1",
        authorized_site_ids=frozenset({"SITE-1"}),
    )

    service.register_asset(
        Asset("PUMP-001", "SITE-1", "pump", "Main water pump")
    )
    service.register_asset(
        Asset("PUMP-002", "SITE-1", "pump", "Backup water pump")
    )

    workflow.start("SESSION-1", access)
    return database, service, workflow, access


def draft(asset_id="PUMP-001", safety=("active_water_leak",)):
    return IncidentDraft(
        session_id="SESSION-1",
        site_id="SITE-1",
        asset_id=asset_id,
        description="Water pump is leaking",
        symptoms=("water pooling",),
        safety_indicators=safety,
    )


def select_asset(workflow, access, asset_id="PUMP-001"):
    workflow.lookup_asset_by_id("SESSION-1", asset_id, access)
    return workflow.select_asset(
        "SESSION-1", "SITE-1", asset_id, access
    )


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


def test_normal_workflow_requires_confirmation_then_creates_ticket(setup):
    database, service, workflow, access = setup
    select_asset(workflow, access)

    preview = workflow.submit_proposal(
        draft(), access, "IDEM-TEXT-001", caller_confirmed=False
    )

    assert preview.status == "confirmation_required"
    assert preview.data["proposal"]["severity"] == "high"
    assert counts(database) == {"incidents": 0, "confirmations": 0}

    submitted = workflow.submit_proposal(
        draft(), access, "IDEM-TEXT-001", caller_confirmed=True
    )

    assert submitted.status == "ticket_created"
    assert submitted.data["incident_id"]
    assert counts(database) == {"incidents": 1, "confirmations": 1}

    assert service.get_session_state(
        "SESSION-1", access
    ) is SessionState.SUBMITTED


def test_ambiguous_search_preserves_candidates_for_explicit_selection(setup):
    _, service, workflow, access = setup

    result = workflow.search_assets_by_clue(
        "SESSION-1", "SITE-1", "pump", access
    )

    assert result.status == "clarification_required"
    assert len(result.data["matches"]) == 2

    selected = workflow.select_asset(
        "SESSION-1", "SITE-1", "PUMP-002", access
    )

    assert selected.status == "ready_for_confirmation"
    assert selected.data["asset_id"] == "PUMP-002"


def test_unknown_asset_can_be_corrected_by_a_second_lookup(setup):
    _, service, workflow, access = setup

    missing = workflow.lookup_asset_by_id(
        "SESSION-1", "PUMP-999", access
    )
    assert missing.status == "clarification_required"

    found = workflow.lookup_asset_by_id(
        "SESSION-1", "PUMP-002", access
    )
    assert found.status == "asset_found"

    selected = workflow.select_asset(
        "SESSION-1", "SITE-1", "PUMP-002", access
    )
    assert selected.status == "ready_for_confirmation"


def test_correction_clears_old_selection(setup):
    _, _, workflow, access = setup
    select_asset(workflow, access, "PUMP-001")

    workflow.lookup_asset_by_id("SESSION-1", "PUMP-002", access)
    workflow.select_asset("SESSION-1", "SITE-1", "PUMP-002", access)

    with pytest.raises(IncidentValidationError, match="must be selected"):
        workflow.submit_proposal(
            draft("PUMP-001"),
            access,
            "IDEM-STALE",
            caller_confirmed=False,
        )


def test_critical_incident_hands_off_without_a_ticket(setup):
    database, service, workflow, access = setup
    critical = draft(asset_id=None, safety=("fire_or_smoke",))

    result = workflow.submit_proposal(
        critical,
        access,
        "IDEM-CRITICAL",
        caller_confirmed=False,
    )

    assert result.status == "safety_handoff"
    assert result.data["incident_created"] is False
    assert counts(database)["incidents"] == 0
    assert service.get_session_state(
        "SESSION-1", access
    ) is SessionState.HANDOFF_REQUIRED


def test_confirmation_flag_must_be_boolean(setup):
    _, _, workflow, access = setup
    select_asset(workflow, access)

    with pytest.raises(TypeError, match="must be a boolean"):
        workflow.submit_proposal(
            draft(), access, "IDEM-INVALID", caller_confirmed="yes"
        )

def test_retry_after_submission_returns_original_ticket(setup):
    database, _, workflow, access = setup
    select_asset(workflow, access)

    first = workflow.submit_proposal(
        draft(), access, "IDEM-WORKFLOW-RETRY", caller_confirmed=True
    )
    second = workflow.submit_proposal(
        draft(), access, "IDEM-WORKFLOW-RETRY", caller_confirmed=True
    )

    assert first.status == "ticket_created"
    assert second.status == "ticket_created"
    assert first.data["incident_id"] == second.data["incident_id"]
    assert second.data["replayed"] is True

    with database.connect() as connection:
        incidents = connection.execute(
            "SELECT COUNT(*) FROM incidents"
        ).fetchone()[0]
        operations = connection.execute(
            "SELECT COUNT(*) FROM tool_operations"
        ).fetchone()[0]
        confirmations = connection.execute(
            "SELECT COUNT(*) FROM confirmation_events"
        ).fetchone()[0]

    assert incidents == 1
    assert operations == 1
    assert confirmations == 1


def test_retry_after_submission_rejects_changed_proposal(setup):
    _, _, workflow, access = setup
    select_asset(workflow, access)

    workflow.submit_proposal(
        draft(), access, "IDEM-WORKFLOW-CONFLICT", caller_confirmed=True
    )

    with pytest.raises(IncidentValidationError, match="different request"):
        workflow.submit_proposal(
            draft().__class__(
                session_id="SESSION-1",
                site_id="SITE-1",
                asset_id="PUMP-001",
                description="A different description",
                symptoms=("water pooling",),
                safety_indicators=("active_water_leak",),
            ),
            access,
            "IDEM-WORKFLOW-CONFLICT",
            caller_confirmed=True,
        )
