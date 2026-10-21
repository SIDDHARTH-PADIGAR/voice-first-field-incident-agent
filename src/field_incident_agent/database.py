"""SQLite persistence foundation for the incident agent."""

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Iterator


SCHEMA_VERSION = 2

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS assets (
        site_id TEXT NOT NULL,
        asset_id TEXT NOT NULL,
        asset_type TEXT NOT NULL CHECK(length(trim(asset_type)) > 0),
        display_name TEXT NOT NULL CHECK(length(trim(display_name)) > 0),
        status TEXT NOT NULL
            CHECK(status IN ('active', 'inactive', 'maintenance')),
        PRIMARY KEY (site_id, asset_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        principal_id TEXT NOT NULL,
        current_state TEXT NOT NULL CHECK(current_state IN (
            'started', 'collecting_details', 'asset_lookup',
            'awaiting_clarification', 'ready_for_confirmation',
            'submitting', 'submitted', 'handoff_required',
            'failed_recoverable', 'failed_final', 'cancelled'
        )),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS incidents (
        incident_id TEXT PRIMARY KEY,
        idempotency_key TEXT NOT NULL UNIQUE,
        session_id TEXT NOT NULL REFERENCES sessions(session_id),
        created_by TEXT NOT NULL,
        site_id TEXT NOT NULL,
        asset_id TEXT NOT NULL,
        description TEXT NOT NULL CHECK(length(trim(description)) > 0),
        symptoms_json TEXT NOT NULL,
        safety_indicators_json TEXT NOT NULL,
        severity TEXT NOT NULL
            CHECK(severity IN ('low', 'medium', 'high', 'critical')),
        status TEXT NOT NULL CHECK(status IN (
            'open', 'acknowledged', 'in_progress', 'resolved', 'cancelled'
        )),
        confirmation_status TEXT NOT NULL
            CHECK(confirmation_status = 'confirmed'),
        confirmed_at TEXT NOT NULL,
        time_noticed TEXT,
        operational_impact TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (site_id, asset_id)
            REFERENCES assets(site_id, asset_id)
            ON UPDATE RESTRICT ON DELETE RESTRICT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tool_operations (
        operation_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(session_id),
        operation_type TEXT NOT NULL
            CHECK(length(trim(operation_type)) > 0),
        idempotency_key TEXT NOT NULL UNIQUE,
        status TEXT NOT NULL
            CHECK(status IN ('pending', 'succeeded', 'failed', 'unknown')),
        request_fingerprint TEXT NOT NULL,
        result_incident_id TEXT REFERENCES incidents(incident_id),
        error_code TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS confirmation_events (
        confirmation_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(session_id),
        principal_id TEXT NOT NULL,
        proposal_fingerprint TEXT NOT NULL,
        confirmed_at TEXT NOT NULL,
        consumed_by_idempotency_key TEXT UNIQUE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_incidents_site_status
    ON incidents(site_id, status)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_operations_session_status
    ON tool_operations(session_id, status)
    """,
)


class Database:
    """File-backed SQLite database with explicit transaction handling."""

    def __init__(self, path: str | Path) -> None:
        raw_path = str(path)

        if not raw_path.strip():
            raise ValueError("Database path cannot be empty.")

        if raw_path == ":memory:":
            raise ValueError(
                "Use a file-backed database so separate connections share "
                "the same persistent state."
            )

        self.path = Path(raw_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            str(self.path),
            timeout=5.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")

        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a transaction; roll back every change if an error occurs."""
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    def initialize(self) -> None:
        """Create the initial schema and record its version."""
        with self.transaction() as connection:
            version = connection.execute(
                "PRAGMA user_version"
            ).fetchone()[0]

            if version not in (0, 1, SCHEMA_VERSION):
                raise RuntimeError(
                    f"Unsupported database schema version: {version}"
                )

            for statement in _SCHEMA_STATEMENTS:
                connection.execute(statement)

            connection.execute(
                f"PRAGMA user_version = {SCHEMA_VERSION}"
            )