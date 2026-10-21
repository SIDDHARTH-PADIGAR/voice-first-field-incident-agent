import json

import pytest

from field_incident_agent.database import Database
from field_incident_agent.incident_service import (
    AuthorizationError,
    IncidentService,
    IncidentValidationError,
    proposal_fingerprint,
)
from field_incident_agent.models import (
    Asset,
    IncidentDraft,
    SessionState,
    Severity,
)
from field_incident_agent.registry import AccessContext
from field_incident_agent.severity_policy import SeverityPolicy


@pytest.fixture
def setup(tmp_path):
    database = Database(tmp_path / "creation.db")
    database.initialize()
    service = IncidentService(database)
    access = AccessContext(
        principal_id="TECH-1",
        authorized_site_ids=frozenset({"SITE-1"}),
    )

    service.register_asset(
        Asset(
            asset_id="PUMP-1",
            site_id="SITE-1",
            asset_type="pump",
            display_name="Main pump",
        )
    )
    service.create_session("SESSION-1", "TECH-1")
    service.transition_session(
        "SESSION-1", SessionState.COLLECTING_DETAILS, access
    )
    service.transition_session(
        "SESSION-1", SessionState.ASSET_LOOKUP, access
    )
    service.transition_session(
        "SESSION-1", SessionState.READY_FOR_CONFIRMATION, access
    )

    return database, service, access


def make_draft(**overrides):
    values = {
        "session_id": "SESSION-1",
        "site_id": "SITE-1",
        "asset_id": "PUMP-1",
        "description": "Pump is leaking",
        "symptoms": ("water pooling",),
        "safety_indicators": ("active_water_leak",),
    }
    values.update(overrides)
    return IncidentDraft(**values)


def confirm(service, draft, access):
    result = service.record_confirmation(draft, access)
    assert result.requires_handoff is False
    assert result.confirmation_id is not None
    return result.confirmation_id


def test_incident_is_persisted_and_session_submitted(setup):
    database, service, access = setup
    draft = make_draft()
    confirmation_id = confirm(service, draft, access)

    result = service.create_incident(
        draft, access, "IDEM-001", confirmation_id
    )

    assert result.status == "succeeded"
    assert result.incident_id is not None
    assert result.operation_id is not None
    assert result.severity is Severity.HIGH
    assert result.replayed is False

    with database.connect() as connection:
        incident = connection.execute(
            "SELECT * FROM incidents WHERE incident_id = ?",
            (result.incident_id,),
        ).fetchone()
        session = connection.execute(
            "SELECT current_state FROM sessions WHERE session_id = ?",
            ("SESSION-1",),
        ).fetchone()
        operation = connection.execute(
            "SELECT status, result_incident_id FROM tool_operations "
            "WHERE operation_id = ?",
            (result.operation_id,),
        ).fetchone()
        confirmation = connection.execute(
            "SELECT consumed_by_idempotency_key FROM confirmation_events "
            "WHERE confirmation_id = ?",
            (confirmation_id,),
        ).fetchone()

    assert incident["status"] == "open"
    assert incident["confirmation_status"] == "confirmed"
    assert "T" in incident["confirmed_at"]
    assert incident["confirmed_at"].endswith("+00:00")
    assert json.loads(incident["symptoms_json"]) == ["water pooling"]
    assert session["current_state"] == SessionState.SUBMITTED.value
    assert operation["status"] == "succeeded"
    assert operation["result_incident_id"] == result.incident_id
    assert confirmation["consumed_by_idempotency_key"] == "IDEM-001"


def test_same_key_and_same_request_returns_existing_incident(setup):
    database, service, access = setup
    draft = make_draft()
    confirmation_id = confirm(service, draft, access)

    first = service.create_incident(
        draft, access, "IDEM-REPLAY", confirmation_id
    )
    second = service.create_incident(
        draft, access, "IDEM-REPLAY", confirmation_id
    )

    assert first.status == second.status == "succeeded"
    assert first.incident_id == second.incident_id
    assert first.operation_id == second.operation_id
    assert second.replayed is True

    with database.connect() as connection:
        incident_count = connection.execute(
            "SELECT COUNT(*) FROM incidents"
        ).fetchone()[0]
        operation_count = connection.execute(
            "SELECT COUNT(*) FROM tool_operations"
        ).fetchone()[0]

    assert incident_count == 1
    assert operation_count == 1


def test_same_key_with_changed_proposal_is_rejected(setup):
    _, service, access = setup
    original = make_draft()
    confirmation_id = confirm(service, original, access)

    service.create_incident(
        original, access, "IDEM-CONFLICT", confirmation_id
    )

    altered = make_draft(description="Pump is making grinding noises")

    with pytest.raises(IncidentValidationError, match="different request"):
        service.create_incident(
            altered, access, "IDEM-CONFLICT", confirmation_id
        )


def test_confirmation_is_required(setup):
    database, service, access = setup
    draft = make_draft()

    with pytest.raises(IncidentValidationError, match="confirmation is required"):
        service.create_incident(
            draft, access, "IDEM-NOCONF", "CONF-NOTFOUND"
        )

    with database.connect() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM incidents"
        ).fetchone()[0]

    assert count == 0


def test_confirmation_cannot_be_reused_with_a_new_key(setup):
    _, service, access = setup
    draft = make_draft()
    confirmation_id = confirm(service, draft, access)

    service.create_incident(
        draft, access, "IDEM-FIRST", confirmation_id
    )

    with pytest.raises(IncidentValidationError, match="already consumed"):
        service.create_incident(
            draft, access, "IDEM-SECOND", confirmation_id
        )


def test_unauthorized_site_cannot_receive_incident(setup):
    database, service, access = setup
    draft = make_draft(site_id="SITE-OTHER")

    with pytest.raises(AuthorizationError, match="not authorized"):
        service.record_confirmation(draft, access)

    with database.connect() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM incidents"
        ).fetchone()[0]

    assert count == 0


def test_critical_case_hands_off_without_creating_incident(setup):
    database, service, access = setup
    draft = make_draft(
        safety_indicators=("fire_or_smoke",),
        site_id=None,
        asset_id=None,
    )

    result = service.create_incident(
        draft, access, "IDEM-CRITICAL", "CONF-NOTNEEDED"
    )

    assert result.status == "handoff_required"
    assert result.severity is Severity.CRITICAL
    assert result.incident_id is None

    with database.connect() as connection:
        incident_count = connection.execute(
            "SELECT COUNT(*) FROM incidents"
        ).fetchone()[0]
        state = connection.execute(
            "SELECT current_state FROM sessions WHERE session_id = ?",
            ("SESSION-1",),
        ).fetchone()["current_state"]

    assert incident_count == 0
    assert state == SessionState.HANDOFF_REQUIRED.value


def test_operation_status_reconciles_unknown_to_succeeded(setup):
    database, service, access = setup
    draft = make_draft()
    confirmation_id = confirm(service, draft, access)

    created = service.create_incident(
        draft, access, "IDEM-RECONCILE", confirmation_id
    )

    with database.transaction() as connection:
        connection.execute(
            "UPDATE tool_operations SET status = 'unknown' "
            "WHERE idempotency_key = ?",
            ("IDEM-RECONCILE",),
        )

    status = service.reconcile_operation_status(
        "SESSION-1", "IDEM-RECONCILE", access
    )

    assert status.status == "succeeded"
    assert status.incident_id == created.incident_id


def test_unknown_operation_without_ticket_is_not_retried(setup):
    database, service, access = setup
    draft = make_draft()
    decision = SeverityPolicy.classify(draft.safety_indicators)
    fingerprint = proposal_fingerprint(draft, access.principal_id, decision)

    with database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO tool_operations (
                operation_id, session_id, operation_type, idempotency_key,
                status, request_fingerprint, result_incident_id, error_code,
                created_at, updated_at
            )
            VALUES (
                'OP-UNKNOWN', 'SESSION-1', 'create_incident',
                'IDEM-UNKNOWN', 'unknown', ?, NULL, NULL, 't0', 't0'
            )
            """,
            (fingerprint,),
        )

    result = service.create_incident(
        draft, access, "IDEM-UNKNOWN", "CONF-UNUSED"
    )

    assert result.status == "unknown"
    assert result.incident_id is None

    status = service.reconcile_operation_status(
        "SESSION-1", "IDEM-UNKNOWN", access
    )
    assert status.status == "unknown"
    assert status.incident_id is None


def test_operation_status_does_not_leak_another_sessions_operation(setup):
    _, service, access = setup
    other_access = AccessContext(
        principal_id="TECH-OTHER",
        authorized_site_ids=frozenset({"SITE-1"}),
    )

    with pytest.raises(AuthorizationError, match="Session unavailable"):
        service.reconcile_operation_status(
            "SESSION-1", "IDEM-UNKNOWN", other_access
        )


def test_malformed_idempotency_key_is_rejected(setup):
    _, service, access = setup
    draft = make_draft()

    with pytest.raises(ValueError, match="idempotency_key"):
        service.create_incident(
            draft, access, "INVALID KEY", "CONF-NOTFOUND"
        )