"""Controlled application services for field incident intake."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
import uuid

from field_incident_agent.database import Database
from field_incident_agent.models import (
    Asset,
    IncidentDraft,
    IncidentStatus,
    SessionState,
    Severity,
)
from field_incident_agent.registry import AccessContext
from field_incident_agent.severity_policy import (
    SeverityDecision,
    SeverityPolicy,
)
from field_incident_agent.state_machine import transition_session_state


_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")


class AuthorizationError(PermissionError):
    """The caller is not authorized to access a session or site."""


class IncidentValidationError(ValueError):
    """The proposed operation violates domain requirements."""


@dataclass(frozen=True, slots=True)
class ConfirmationResult:
    confirmation_id: str | None
    severity_decision: SeverityDecision
    requires_handoff: bool


@dataclass(frozen=True, slots=True)
class IncidentCreationResult:
    status: str
    incident_id: str | None
    operation_id: str | None
    severity: Severity
    replayed: bool = False


@dataclass(frozen=True, slots=True)
class OperationStatusResult:
    status: str
    operation_id: str | None
    incident_id: str | None


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_identifier(value: str, field: str) -> None:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise ValueError(f"{field} is not a valid identifier.")


def proposal_fingerprint(
    draft: IncidentDraft,
    principal_id: str,
    decision: SeverityDecision,
) -> str:
    """Fingerprint the exact incident proposal and policy outcome."""
    payload = {
        "policy_version": 1,
        "session_id": draft.session_id,
        "principal_id": principal_id,
        "site_id": draft.site_id,
        "asset_id": draft.asset_id,
        "description": draft.description,
        "symptoms": list(draft.symptoms),
        "safety_indicators": list(decision.indicators),
        "time_noticed": draft.time_noticed,
        "operational_impact": draft.operational_impact,
        "severity": decision.severity.value,
        "severity_reason": decision.reason,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class IncidentService:
    """Controlled persistence operations; never expose the database to an LLM."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def register_asset(self, asset: Asset) -> None:
        """Register an approved asset through trusted setup/admin code."""
        if not isinstance(asset, Asset):
            raise TypeError("asset must be an Asset.")

        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO assets (
                    site_id, asset_id, asset_type, display_name, status
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    asset.site_id,
                    asset.asset_id,
                    asset.asset_type,
                    asset.display_name,
                    asset.status.value,
                ),
            )

    def create_session(
        self,
        session_id: str,
        principal_id: str,
    ) -> None:
        """Create a session for an identity established by trusted backend code."""
        _validate_identifier(session_id, "session_id")
        _validate_identifier(principal_id, "principal_id")
        now = _timestamp()

        with self.database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    session_id, principal_id, current_state,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    principal_id,
                    SessionState.STARTED.value,
                    now,
                    now,
                ),
            )

    @staticmethod
    def _require_session(connection, session_id, access):
        """Return an owned session without revealing whether others exist."""
        if not isinstance(access, AccessContext):
            raise TypeError("access must be a trusted AccessContext.")

        _validate_identifier(session_id, "session_id")
        row = connection.execute(
            """
            SELECT session_id, principal_id, current_state
            FROM sessions WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()

        if row is None or row["principal_id"] != access.principal_id:
            raise AuthorizationError("Session unavailable.")

        return row

    @staticmethod
    def _validate_site_and_asset(connection, draft, access) -> None:
        if draft.site_id is None or draft.asset_id is None:
            raise IncidentValidationError(
                "A confirmed proposal requires a site and resolved asset."
            )

        if draft.site_id not in access.authorized_site_ids:
            raise AuthorizationError("Caller is not authorized for this site.")

        asset = connection.execute(
            """
            SELECT status FROM assets
            WHERE site_id = ? AND asset_id = ?
            """,
            (draft.site_id, draft.asset_id),
        ).fetchone()

        if asset is None:
            raise IncidentValidationError(
                "Asset was not found in the authorized site."
            )

        if asset["status"] == "inactive":
            raise IncidentValidationError(
                "An inactive asset cannot be submitted through this workflow."
            )

    def transition_session(
        self,
        session_id: str,
        target: SessionState,
        access: AccessContext,
    ) -> SessionState:
        """Apply an authorized and legal non-submission transition."""
        if target in {
            SessionState.SUBMITTING,
            SessionState.SUBMITTED,
        }:
            raise IncidentValidationError(
                "Submission states can only be changed by the incident writer."
            )

        with self.database.transaction() as connection:
            session = self._require_session(connection, session_id, access)
            current = SessionState(session["current_state"])

            try:
                next_state = transition_session_state(current, target)
            except ValueError as exc:
                raise IncidentValidationError(str(exc)) from exc

            connection.execute(
                """
                UPDATE sessions
                SET current_state = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (next_state.value, _timestamp(), session_id),
            )
            return next_state

    def record_confirmation(
        self,
        draft: IncidentDraft,
        access: AccessContext,
    ) -> ConfirmationResult:
        """Record a trusted, explicit caller confirmation of one proposal."""
        if not isinstance(draft, IncidentDraft):
            raise TypeError("draft must be an IncidentDraft.")

        if not isinstance(access, AccessContext):
            raise TypeError("access must be a trusted AccessContext.")

        decision = SeverityPolicy.classify(draft.safety_indicators)

        with self.database.transaction() as connection:
            session = self._require_session(
                connection, draft.session_id, access
            )
            current = SessionState(session["current_state"])

            if decision.requires_handoff:
                try:
                    next_state = transition_session_state(
                        current,
                        SessionState.HANDOFF_REQUIRED,
                    )
                except ValueError as exc:
                    raise IncidentValidationError(str(exc)) from exc

                connection.execute(
                    """
                    UPDATE sessions
                    SET current_state = ?, updated_at = ?
                    WHERE session_id = ?
                    """,
                    (next_state.value, _timestamp(), draft.session_id),
                )

                return ConfirmationResult(
                    confirmation_id=None,
                    severity_decision=decision,
                    requires_handoff=True,
                )

            if current is not SessionState.READY_FOR_CONFIRMATION:
                raise IncidentValidationError(
                    "Session is not ready for caller confirmation."
                )

            self._validate_site_and_asset(connection, draft, access)
            fingerprint = proposal_fingerprint(
                draft, access.principal_id, decision
            )
            confirmation_id = f"CONF-{uuid.uuid4().hex}"

            connection.execute(
                """
                INSERT INTO confirmation_events (
                    confirmation_id, session_id, principal_id,
                    proposal_fingerprint, confirmed_at,
                    consumed_by_idempotency_key
                )
                VALUES (?, ?, ?, ?, ?, NULL)
                """,
                (
                    confirmation_id,
                    draft.session_id,
                    access.principal_id,
                    fingerprint,
                    _timestamp(),
                ),
            )

            return ConfirmationResult(
                confirmation_id=confirmation_id,
                severity_decision=decision,
                requires_handoff=False,
            )

    def create_incident(
        self,
        draft: IncidentDraft,
        access: AccessContext,
        idempotency_key: str,
        confirmation_id: str,
    ) -> IncidentCreationResult:
        """Create one confirmed incident in a single database transaction.

        Replays of the same key and proposal return the persisted result.
        Reusing a key for a different proposal is rejected.
        Unknown outcomes are never blindly retried.
        """
        if not isinstance(draft, IncidentDraft):
            raise TypeError("draft must be an IncidentDraft.")

        if not isinstance(access, AccessContext):
            raise TypeError("access must be a trusted AccessContext.")

        _validate_identifier(idempotency_key, "idempotency_key")
        _validate_identifier(confirmation_id, "confirmation_id")

        decision = SeverityPolicy.classify(draft.safety_indicators)
        fingerprint = proposal_fingerprint(
            draft, access.principal_id, decision
        )

        with self.database.transaction() as connection:
            session = self._require_session(
                connection, draft.session_id, access
            )

            # Resolve replays before checking the current session state.
            # A successful replay necessarily sees a SUBMITTED session.
            operation = connection.execute(
                """
                SELECT operation_id, session_id, operation_type,
                       status, request_fingerprint, result_incident_id
                FROM tool_operations
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()

            if operation is not None:
                if (
                    operation["session_id"] != draft.session_id
                    or operation["operation_type"] != "create_incident"
                    or operation["request_fingerprint"] != fingerprint
                ):
                    raise IncidentValidationError(
                        "Idempotency key conflicts with a different request."
                    )

                operation_id = operation["operation_id"]
                status = operation["status"]
                incident_id = operation["result_incident_id"]

                if status == "succeeded":
                    incident = connection.execute(
                        """
                        SELECT incident_id FROM incidents
                        WHERE incident_id = ? AND idempotency_key = ?
                        """,
                        (incident_id, idempotency_key),
                    ).fetchone()

                    if incident is not None:
                        return IncidentCreationResult(
                            status="succeeded",
                            incident_id=incident["incident_id"],
                            operation_id=operation_id,
                            severity=decision.severity,
                            replayed=True,
                        )

                    connection.execute(
                        """
                        UPDATE tool_operations
                        SET status = 'unknown', updated_at = ?
                        WHERE operation_id = ?
                        """,
                        (_timestamp(), operation_id),
                    )
                    return IncidentCreationResult(
                        status="unknown",
                        incident_id=None,
                        operation_id=operation_id,
                        severity=decision.severity,
                        replayed=True,
                    )

                if status in {"pending", "unknown"}:
                    # The durable record must be checked before any retry.
                    incident = connection.execute(
                        """
                        SELECT incident_id FROM incidents
                        WHERE idempotency_key = ?
                        """,
                        (idempotency_key,),
                    ).fetchone()

                    if incident is not None:
                        connection.execute(
                            """
                            UPDATE tool_operations
                            SET status = 'succeeded',
                                result_incident_id = ?,
                                updated_at = ?
                            WHERE operation_id = ?
                            """,
                            (
                                incident["incident_id"],
                                _timestamp(),
                                operation_id,
                            ),
                        )
                        return IncidentCreationResult(
                            status="succeeded",
                            incident_id=incident["incident_id"],
                            operation_id=operation_id,
                            severity=decision.severity,
                            replayed=True,
                        )

                    # Do not retry while the outcome remains unknown.
                    return IncidentCreationResult(
                        status="unknown",
                        incident_id=None,
                        operation_id=operation_id,
                        severity=decision.severity,
                        replayed=True,
                    )

                return IncidentCreationResult(
                    status=status,
                    incident_id=incident_id,
                    operation_id=operation_id,
                    severity=decision.severity,
                    replayed=True,
                )

            # Detect a record without its operation metadata. Never assume
            # that this means the previous mutation failed.
            orphan = connection.execute(
                """
                SELECT incident_id FROM incidents
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()

            if orphan is not None:
                raise IncidentValidationError(
                    "Incident exists without matching operation metadata; "
                    "manual reconciliation is required."
                )

            current = SessionState(session["current_state"])

            if decision.requires_handoff:
                if current is not SessionState.HANDOFF_REQUIRED:
                    try:
                        next_state = transition_session_state(
                            current,
                            SessionState.HANDOFF_REQUIRED,
                        )
                    except ValueError as exc:
                        raise IncidentValidationError(str(exc)) from exc

                    connection.execute(
                        """
                        UPDATE sessions
                        SET current_state = ?, updated_at = ?
                        WHERE session_id = ?
                        """,
                        (
                            next_state.value,
                            _timestamp(),
                            draft.session_id,
                        ),
                    )

                return IncidentCreationResult(
                    status="handoff_required",
                    incident_id=None,
                    operation_id=None,
                    severity=decision.severity,
                )

            # Recheck authorization and asset status at write time.
            self._validate_site_and_asset(connection, draft, access)

            confirmation = connection.execute(
                """
                SELECT proposal_fingerprint, consumed_by_idempotency_key
                FROM confirmation_events
                WHERE confirmation_id = ?
                  AND session_id = ?
                  AND principal_id = ?
                """,
                (
                    confirmation_id,
                    draft.session_id,
                    access.principal_id,
                ),
            ).fetchone()

            if confirmation is None:
                raise IncidentValidationError(
                    "A matching caller confirmation is required."
                )

            if confirmation["proposal_fingerprint"] != fingerprint:
                raise IncidentValidationError(
                    "The confirmed proposal differs from the submitted proposal."
                )

            if confirmation["consumed_by_idempotency_key"] is not None:
                raise IncidentValidationError(
                    "Confirmation was already consumed or requires reconciliation."
                )

            if current is not SessionState.READY_FOR_CONFIRMATION:
                raise IncidentValidationError(
                    "Session is not ready for incident submission."
                )

            now = _timestamp()
            operation_id = f"OP-{uuid.uuid4().hex}"
            incident_id = f"INC-{uuid.uuid4().hex}"

            # BEGIN IMMEDIATE serializes SQLite writers. Operation, ticket,
            # confirmation consumption, and final state commit together.
            connection.execute(
                """
                INSERT INTO tool_operations (
                    operation_id, session_id, operation_type,
                    idempotency_key, status, request_fingerprint,
                    result_incident_id, error_code, created_at, updated_at
                )
                VALUES (?, ?, 'create_incident', ?, 'pending',
                        ?, NULL, NULL, ?, ?)
                """,
                (
                    operation_id,
                    draft.session_id,
                    idempotency_key,
                    fingerprint,
                    now,
                    now,
                ),
            )

            connection.execute(
                """
                UPDATE sessions
                SET current_state = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (SessionState.SUBMITTING.value, now, draft.session_id),
            )

            consumed = connection.execute(
                """
                UPDATE confirmation_events
                SET consumed_by_idempotency_key = ?
                WHERE confirmation_id = ?
                  AND consumed_by_idempotency_key IS NULL
                """,
                (idempotency_key, confirmation_id),
            )

            if consumed.rowcount != 1:
                raise IncidentValidationError(
                    "Confirmation could not be consumed safely."
                )

            connection.execute(
                """
                INSERT INTO incidents (
                    incident_id, idempotency_key, session_id, created_by,
                    site_id, asset_id, description, symptoms_json,
                    safety_indicators_json, severity, status,
                    confirmation_status, confirmed_at, time_noticed,
                    operational_impact, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'confirmed',
                        ?, ?, ?, ?)
                """,
                (
                    incident_id,
                    idempotency_key,
                    draft.session_id,
                    access.principal_id,
                    draft.site_id,
                    draft.asset_id,
                    draft.description,
                    json.dumps(list(draft.symptoms)),
                    json.dumps(list(decision.indicators)),
                    decision.severity.value,
                    IncidentStatus.OPEN.value,
                    confirmation["proposal_fingerprint"],
                    draft.time_noticed,
                    draft.operational_impact,
                    now,
                ),
            )

            connection.execute(
                """
                UPDATE tool_operations
                SET status = 'succeeded', result_incident_id = ?,
                    updated_at = ?
                WHERE operation_id = ?
                """,
                (incident_id, _timestamp(), operation_id),
            )

            connection.execute(
                """
                UPDATE sessions
                SET current_state = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (SessionState.SUBMITTED.value, _timestamp(), draft.session_id),
            )

            return IncidentCreationResult(
                status="succeeded",
                incident_id=incident_id,
                operation_id=operation_id,
                severity=decision.severity,
            )

    def reconcile_operation_status(
        self,
        session_id: str,
        idempotency_key: str,
        access: AccessContext,
    ) -> OperationStatusResult:
        """Reconcile an operation against persisted incident state.

        Unknown outcomes without a corresponding persisted incident remain
        unknown. This method never retries a mutation.
        """
        _validate_identifier(idempotency_key, "idempotency_key")

        with self.database.transaction() as connection:
            self._require_session(connection, session_id, access)

            operation = connection.execute(
                """
                SELECT operation_id, session_id, status, result_incident_id
                FROM tool_operations
                WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()

            if operation is None:
                return OperationStatusResult(
                    status="not_found",
                    operation_id=None,
                    incident_id=None,
                )

            if operation["session_id"] != session_id:
                raise AuthorizationError("Operation unavailable.")

            status = operation["status"]
            incident_id = operation["result_incident_id"]

            if status in {"pending", "unknown"}:
                incident = connection.execute(
                    """
                    SELECT incident_id FROM incidents
                    WHERE idempotency_key = ?
                    """,
                    (idempotency_key,),
                ).fetchone()

                if incident is not None:
                    incident_id = incident["incident_id"]
                    status = "succeeded"
                    connection.execute(
                        """
                        UPDATE tool_operations
                        SET status = 'succeeded', result_incident_id = ?,
                            updated_at = ?
                        WHERE operation_id = ?
                        """,
                        (incident_id, _timestamp(), operation["operation_id"]),
                    )
                else:
                    status = "unknown"

            return OperationStatusResult(
                status=status,
                operation_id=operation["operation_id"],
                incident_id=incident_id,
            )