import sqlite3

import pytest

from field_incident_agent.database import Database, SCHEMA_VERSION
from field_incident_agent.incident_service import (
    AuthorizationError,
    IncidentService,
    IncidentValidationError,
)
from field_incident_agent.models import (
    Asset,
    AssetStatus,
    IncidentDraft,
    SessionState,
    Severity,
)
from field_incident_agent.registry import AccessContext


@pytest.fixture
def setup(tmp_path):
    database = Database(tmp_path / "service.db")
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


def test_confirmation_event_is_persisted_and_bound_to_proposal(setup):
    database, service, access = setup
    draft = make_draft()

    result = service.record_confirmation(draft, access)

    assert result.confirmation_id is not None
    assert result.requires_handoff is False
    assert result.severity_decision.severity is Severity.HIGH

    with database.connect() as connection:
        row = connection.execute(
            """
            SELECT session_id, principal_id, proposal_fingerprint,
                   confirmed_at, consumed_by_idempotency_key
            FROM confirmation_events WHERE confirmation_id = ?
            """,
            (result.confirmation_id,),
        ).fetchone()

    assert row["session_id"] == "SESSION-1"
    assert row["principal_id"] == "TECH-1"
    assert len(row["proposal_fingerprint"]) == 64
    assert row["confirmed_at"]
    assert row["consumed_by_idempotency_key"] is None


def test_critical_indicator_routes_to_handoff_without_confirmation(setup):
    database, service, access = setup

    service.create_session("SESSION-CRIT", "TECH-1")
    service.transition_session(
        "SESSION-CRIT", SessionState.COLLECTING_DETAILS, access
    )

    result = service.record_confirmation(
        make_draft(
            session_id="SESSION-CRIT",
            safety_indicators=("fire_or_smoke",),
            site_id=None,
            asset_id=None,
        ),
        access,
    )

    assert result.requires_handoff is True
    assert result.confirmation_id is None
    assert result.severity_decision.severity is Severity.CRITICAL

    with database.connect() as connection:
        state = connection.execute(
            "SELECT current_state FROM sessions WHERE session_id = ?",
            ("SESSION-CRIT",),
        ).fetchone()["current_state"]
        confirmations = connection.execute(
            "SELECT COUNT(*) FROM confirmation_events"
        ).fetchone()[0]

    assert state == SessionState.HANDOFF_REQUIRED.value
    assert confirmations == 0


def test_unknown_safety_indicator_is_rejected(setup):
    _, service, access = setup

    with pytest.raises(ValueError, match="Unsupported safety indicators"):
        service.record_confirmation(
            make_draft(safety_indicators=("probably_safe",)),
            access,
        )


def test_site_and_asset_are_required_for_normal_confirmation(setup):
    _, service, access = setup

    with pytest.raises(IncidentValidationError, match="site and resolved asset"):
        service.record_confirmation(
            make_draft(site_id=None, asset_id=None),
            access,
        )


def test_caller_cannot_confirm_for_unauthorized_site(setup):
    _, service, access = setup

    with pytest.raises(AuthorizationError, match="not authorized"):
        service.record_confirmation(
            make_draft(site_id="SITE-OTHER"),
            access,
        )


def test_inactive_asset_cannot_be_confirmed(setup):
    _, service, access = setup

    service.register_asset(
        Asset(
            asset_id="PUMP-OLD",
            site_id="SITE-1",
            asset_type="pump",
            display_name="Retired pump",
            status=AssetStatus.INACTIVE,
        )
    )

    with pytest.raises(IncidentValidationError, match="inactive asset"):
        service.record_confirmation(
            make_draft(asset_id="PUMP-OLD"),
            access,
        )


def test_session_must_be_ready_for_normal_confirmation(setup):
    _, service, access = setup
    service.create_session("SESSION-NEW", "TECH-1")

    with pytest.raises(IncidentValidationError, match="not ready"):
        service.record_confirmation(
            make_draft(session_id="SESSION-NEW"),
            access,
        )


def test_caller_cannot_access_another_principals_session(setup):
    _, service, _ = setup
    other_access = AccessContext(
        principal_id="TECH-OTHER",
        authorized_site_ids=frozenset({"SITE-1"}),
    )

    with pytest.raises(AuthorizationError, match="Session unavailable"):
        service.transition_session(
            "SESSION-1",
            SessionState.CANCELLED,
            other_access,
        )


@pytest.mark.parametrize(
    "target",
    [SessionState.SUBMITTING, SessionState.SUBMITTED],
)
def test_submission_states_cannot_be_set_directly(setup, target):
    _, service, access = setup

    with pytest.raises(IncidentValidationError, match="only be changed"):
        service.transition_session("SESSION-1", target, access)


def test_duplicate_asset_registration_is_rejected(setup):
    _, service, _ = setup

    with pytest.raises(sqlite3.IntegrityError):
        service.register_asset(
            Asset(
                asset_id="PUMP-1",
                site_id="SITE-1",
                asset_type="pump",
                display_name="Duplicate pump",
            )
        )


def test_database_schema_is_version_two(setup):
    database, _, _ = setup

    with database.connect() as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert version == SCHEMA_VERSION == 2
    assert "confirmation_events" in tables