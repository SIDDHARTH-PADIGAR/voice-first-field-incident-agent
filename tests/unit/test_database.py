import sqlite3

import pytest

from field_incident_agent.database import Database, SCHEMA_VERSION


@pytest.fixture
def database(tmp_path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    return db


def _seed_minimum(connection):
    connection.execute(
        """
        INSERT INTO sessions
            (session_id, principal_id, current_state, created_at, updated_at)
        VALUES ('SESSION-1', 'TECH-1', 'started', 't0', 't0')
        """
    )
    connection.execute(
        """
        INSERT INTO assets
            (site_id, asset_id, asset_type, display_name, status)
        VALUES ('SITE-1', 'PUMP-1', 'pump', 'Main pump', 'active')
        """
    )


def _insert_incident(
    connection,
    incident_id,
    idempotency_key,
    *,
    site_id="SITE-1",
    asset_id="PUMP-1",
    confirmation_status="confirmed",
):
    connection.execute(
        """
        INSERT INTO incidents (
            incident_id, idempotency_key, session_id, created_by,
            site_id, asset_id, description, symptoms_json,
            safety_indicators_json, severity, status,
            confirmation_status, confirmed_at, time_noticed,
            operational_impact, created_at
        )
        VALUES (?, ?, 'SESSION-1', 'TECH-1', ?, ?, 'Pump is leaking',
                '[]', '[]', 'medium', 'open', ?, 't1', NULL, NULL, 't1')
        """,
        (
            incident_id,
            idempotency_key,
            site_id,
            asset_id,
            confirmation_status,
        ),
    )


def test_initialization_is_repeatable(database):
    database.initialize()

    with database.connect() as connection:
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        version = connection.execute("PRAGMA user_version").fetchone()[0]

    assert {"assets", "sessions", "incidents", "tool_operations"} <= tables
    assert version == SCHEMA_VERSION


def test_foreign_keys_are_enabled(database):
    with database.connect() as connection:
        enabled = connection.execute("PRAGMA foreign_keys").fetchone()[0]

    assert enabled == 1


def test_failed_transaction_rolls_back_all_changes(database):
    with pytest.raises(RuntimeError, match="force rollback"):
        with database.transaction() as connection:
            _seed_minimum(connection)
            raise RuntimeError("force rollback")

    with database.connect() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM sessions"
        ).fetchone()[0]

    assert count == 0


def test_asset_id_can_repeat_across_different_sites(database):
    with database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO assets
                (site_id, asset_id, asset_type, display_name, status)
            VALUES ('SITE-1', 'PUMP-1', 'pump', 'Pump one', 'active')
            """
        )
        connection.execute(
            """
            INSERT INTO assets
                (site_id, asset_id, asset_type, display_name, status)
            VALUES ('SITE-2', 'PUMP-1', 'pump', 'Pump two', 'active')
            """
        )

    with database.connect() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM assets WHERE asset_id = 'PUMP-1'"
        ).fetchone()[0]

    assert count == 2


def test_duplicate_asset_id_within_same_site_is_rejected(database):
    with database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO assets
                (site_id, asset_id, asset_type, display_name, status)
            VALUES ('SITE-1', 'PUMP-1', 'pump', 'Pump one', 'active')
            """
        )

    with pytest.raises(sqlite3.IntegrityError):
        with database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO assets
                    (site_id, asset_id, asset_type, display_name, status)
                VALUES ('SITE-1', 'PUMP-1', 'pump', 'Duplicate', 'active')
                """
            )


def test_incident_cannot_reference_an_asset_at_another_site(database):
    with database.transaction() as connection:
        _seed_minimum(connection)

    with pytest.raises(sqlite3.IntegrityError):
        with database.transaction() as connection:
            _insert_incident(
                connection,
                "INC-1",
                "IDEM-1",
                site_id="SITE-2",
            )


def test_incident_idempotency_key_must_be_unique(database):
    with database.transaction() as connection:
        _seed_minimum(connection)
        _insert_incident(connection, "INC-1", "IDEM-1")

    with pytest.raises(sqlite3.IntegrityError):
        with database.transaction() as connection:
            _insert_incident(connection, "INC-2", "IDEM-1")


def test_database_rejects_unconfirmed_incident(database):
    with database.transaction() as connection:
        _seed_minimum(connection)

    with pytest.raises(sqlite3.IntegrityError):
        with database.transaction() as connection:
            _insert_incident(
                connection,
                "INC-1",
                "IDEM-1",
                confirmation_status="pending",
            )


def test_database_rejects_unknown_schema_versions(tmp_path):
    db = Database(tmp_path / "future.db")

    with db.connect() as connection:
        connection.execute("PRAGMA user_version = 99")

    with pytest.raises(RuntimeError, match="Unsupported database schema"):
        db.initialize()