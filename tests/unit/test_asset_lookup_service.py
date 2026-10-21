import pytest

from field_incident_agent.database import Database
from field_incident_agent.incident_service import (
    AuthorizationError,
    IncidentService,
    IncidentValidationError,
)
from field_incident_agent.models import Asset, SessionState
from field_incident_agent.registry import AccessContext, LookupStatus


@pytest.fixture
def setup(tmp_path):
    database = Database(tmp_path / "lookup.db")
    database.initialize()
    service = IncidentService(database)

    access = AccessContext(
        principal_id="TECH-1",
        authorized_site_ids=frozenset({"SITE-1"}),
    )

    for asset in (
        Asset("PUMP-001", "SITE-1", "pump", "Main water pump"),
        Asset("PUMP-002", "SITE-1", "pump", "Backup water pump"),
        Asset("PUMP-001", "SITE-2", "pump", "Restricted water pump"),
        Asset("FAN-001", "SITE-2", "fan", "Exhaust fan"),
    ):
        service.register_asset(asset)

    service.create_session("SESSION-1", "TECH-1")
    service.transition_session(
        "SESSION-1", SessionState.COLLECTING_DETAILS, access
    )
    service.transition_session(
        "SESSION-1", SessionState.ASSET_LOOKUP, access
    )

    return database, service, access


def test_exact_lookup_only_returns_authorized_site_match(setup):
    _, service, access = setup

    result = service.lookup_asset_by_id(
        "SESSION-1", "PUMP-001", access
    )

    assert result.status is LookupStatus.FOUND
    assert len(result.matches) == 1
    assert result.matches[0].site_id == "SITE-1"


def test_unknown_asset_returns_not_found(setup):
    _, service, access = setup

    result = service.lookup_asset_by_id(
        "SESSION-1", "PUMP-999", access
    )

    assert result.status is LookupStatus.NOT_FOUND
    assert result.matches == ()


def test_search_reports_ambiguous_matches(setup):
    _, service, access = setup

    result = service.search_assets_by_clue(
        "SESSION-1", "SITE-1", "water pump", access
    )

    assert result.status is LookupStatus.AMBIGUOUS
    assert {asset.asset_id for asset in result.matches} == {
        "PUMP-001", "PUMP-002"
    }


def test_unauthorized_site_search_is_rejected(setup):
    _, service, access = setup

    with pytest.raises(AuthorizationError, match="not authorized"):
        service.search_assets_by_clue(
            "SESSION-1", "SITE-2", "pump", access
        )


def test_selection_advances_session_to_confirmation(setup):
    database, service, access = setup

    service.lookup_asset_by_id("SESSION-1", "PUMP-001", access)
    selected = service.select_asset(
        "SESSION-1", "SITE-1", "PUMP-001", access
    )

    assert selected.asset_id == "PUMP-001"

    with database.connect() as connection:
        state = connection.execute(
            "SELECT current_state FROM sessions WHERE session_id = ?",
            ("SESSION-1",),
        ).fetchone()["current_state"]

    assert state == SessionState.READY_FOR_CONFIRMATION.value


def test_selection_must_come_from_latest_lookup(setup):
    _, service, access = setup

    service.lookup_asset_by_id("SESSION-1", "PUMP-001", access)
    service.lookup_asset_by_id("SESSION-1", "PUMP-002", access)

    with pytest.raises(IncidentValidationError, match="latest authorized lookup"):
        service.select_asset(
            "SESSION-1", "SITE-1", "PUMP-001", access
        )


def test_session_cannot_skip_selection_and_enter_confirmation(setup):
    _, service, access = setup

    with pytest.raises(IncidentValidationError, match="asset selection is required"):
        service.transition_session(
            "SESSION-1",
            SessionState.READY_FOR_CONFIRMATION,
            access,
        )


def test_lookup_requires_asset_lookup_state(setup):
    _, service, access = setup
    service.create_session("SESSION-2", "TECH-1")

    with pytest.raises(IncidentValidationError, match="asset_lookup state"):
        service.lookup_asset_by_id(
            "SESSION-2", "PUMP-001", access
        )