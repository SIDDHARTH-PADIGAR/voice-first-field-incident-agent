from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import sqlite3

import pytest

from field_incident_agent.database import Database
from field_incident_agent.incident_service import (
    IncidentService,
    IncidentValidationError,
)
from field_incident_agent.models import Asset, IncidentDraft, SessionState
from field_incident_agent.registry import AccessContext


@pytest.fixture
def setup(tmp_path):
    database = Database(tmp_path / "reliability.db")
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

    draft = IncidentDraft(
        session_id="SESSION-1",
        site_id="SITE-1",
        asset_id="PUMP-1",
        description="Pump is leaking",
        symptoms=("water pooling",),
        safety_indicators=("active_water_leak",),
    )

    confirmation = service.record_confirmation(draft, access)
    assert confirmation.confirmation_id is not None

    return database, service, access, draft, confirmation.confirmation_id


def _counts(database):
    with database.connect() as connection:
        return {
            "incidents": connection.execute(
                "SELECT COUNT(*) FROM incidents"
            ).fetchone()[0],
            "operations": connection.execute(
                "SELECT COUNT(*) FROM tool_operations"
            ).fetchone()[0],
        }


def test_concurrent_duplicate_requests_create_exactly_one_ticket(setup):
    database, service, access, draft, confirmation_id = setup
    barrier = Barrier(2)

    def submit():
        barrier.wait(timeout=5)
        return service.create_incident(
            draft, access, "IDEM-CONCURRENT", confirmation_id
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(submit)
        second = pool.submit(submit)
        results = [first.result(timeout=10), second.result(timeout=10)]

    assert all(result.status == "succeeded" for result in results)
    assert results[0].incident_id == results[1].incident_id
    assert sorted(result.replayed for result in results) == [False, True]
    assert _counts(database) == {"incidents": 1, "operations": 1}


def test_concurrent_different_keys_cannot_consume_one_confirmation_twice(
    setup,
):
    database, service, access, draft, confirmation_id = setup
    barrier = Barrier(2)

    def submit(key):
        barrier.wait(timeout=5)
        try:
            return service.create_incident(
                draft, access, key, confirmation_id
            )
        except IncidentValidationError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(submit, "IDEM-RACE-A")
        second = pool.submit(submit, "IDEM-RACE-B")
        results = [first.result(timeout=10), second.result(timeout=10)]

    successes = [
        result for result in results
        if not isinstance(result, IncidentValidationError)
    ]
    rejected = [
        result for result in results
        if isinstance(result, IncidentValidationError)
    ]

    assert len(successes) == 1
    assert len(rejected) == 1
    assert "confirmation" in str(rejected[0]).lower()
    assert _counts(database) == {"incidents": 1, "operations": 1}


def test_failed_incident_insert_rolls_back_the_entire_submission(setup):
    database, service, access, draft, confirmation_id = setup

    # Inject a deterministic persistence failure after the operation record
    # and confirmation consumption would otherwise have been written.
    with database.transaction() as connection:
        connection.execute(
            """
            CREATE TRIGGER force_incident_insert_failure
            BEFORE INSERT ON incidents
            BEGIN
                SELECT RAISE(ABORT, 'simulated incident persistence failure');
            END
            """
        )

    with pytest.raises(
        sqlite3.IntegrityError,
        match="simulated incident persistence failure",
    ):
        service.create_incident(
            draft, access, "IDEM-ROLLBACK", confirmation_id
        )

    assert _counts(database) == {"incidents": 0, "operations": 0}

    with database.connect() as connection:
        session = connection.execute(
            "SELECT current_state FROM sessions WHERE session_id = ?",
            ("SESSION-1",),
        ).fetchone()

        confirmation = connection.execute(
            """
            SELECT consumed_by_idempotency_key
            FROM confirmation_events WHERE confirmation_id = ?
            """,
            (confirmation_id,),
        ).fetchone()

    assert session["current_state"] == SessionState.READY_FOR_CONFIRMATION.value
    assert confirmation["consumed_by_idempotency_key"] is None


def test_database_can_reinitialize_after_schema_version_one(setup):
    database, _, _, _, _ = setup

    # Simulate the table/version arrangement of the earlier schema.
    with database.transaction() as connection:
        connection.execute("DROP TABLE confirmation_events")
        connection.execute("PRAGMA user_version = 1")

    database.initialize()

    with database.connect() as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        confirmation_table = connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name = 'confirmation_events'
            """
        ).fetchone()

    assert version == 2
    assert confirmation_table is not None