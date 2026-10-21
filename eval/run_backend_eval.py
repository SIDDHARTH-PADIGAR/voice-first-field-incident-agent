"""Deterministic backend replay. Does not invoke or evaluate an LLM."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from field_incident_agent.database import Database
from field_incident_agent.incident_service import (
    AuthorizationError,
    IncidentService,
    IncidentValidationError,
    IncidentCreationResult,
    ConfirmationResult,
    OperationStatusResult,
    proposal_fingerprint,
)
from field_incident_agent.models import (
    Asset,
    AssetStatus,
    IncidentDraft,
    SessionState,
)
from field_incident_agent.registry import (
    AccessContext,
    AssetLookupResult,
    LookupStatus,
)
from field_incident_agent.severity_policy import SeverityPolicy


CATALOG_PATH = ROOT / "eval" / "scenarios" / "backend_v1.json"
REPORT_PATH = ROOT / "eval" / "results" / "backend_v1_report.json"

# These assertions require evaluating conversational output or interpreting
# the caller's natural-language input. They are explicitly deferred.
DEFERRED_ASSERTIONS = {
    "BACKEND-005": {
        "no_false_claim_of_human_contact":
            "Requires evaluation of the agent's spoken/text response."
    },
    "BACKEND-010": {
        "no_false_success_message":
            "Requires evaluation of the agent's spoken/text response."
    },
    "BACKEND-012": {
        "caller_text_cannot_expand_permissions":
            "Requires testing the LLM/tool orchestrator, which is not implemented."
    },
}


def summarize_result(value):
    """Return a compact, JSON-safe representation of a backend result."""
    if isinstance(value, AssetLookupResult):
        return {
            "status": value.status.value,
            "matches": [
                {"site_id": item.site_id, "asset_id": item.asset_id}
                for item in value.matches
            ],
        }

    if isinstance(value, ConfirmationResult):
        return {
            "confirmation_id": value.confirmation_id,
            "severity": value.severity_decision.severity.value,
            "requires_handoff": value.requires_handoff,
        }

    if isinstance(value, IncidentCreationResult):
        return {
            "status": value.status,
            "incident_id": value.incident_id,
            "operation_id": value.operation_id,
            "severity": value.severity.value,
            "replayed": value.replayed,
        }

    if isinstance(value, OperationStatusResult):
        return {
            "status": value.status,
            "operation_id": value.operation_id,
            "incident_id": value.incident_id,
        }

    if isinstance(value, Asset):
        return {
            "site_id": value.site_id,
            "asset_id": value.asset_id,
            "status": value.status.value,
        }

    return value


class ReplayCase:
    def __init__(self, scenario, catalog, database_path):
        self.scenario = scenario
        self.catalog = catalog
        self.database = Database(database_path)
        self.database.initialize()
        self.service = IncidentService(self.database)
        self.trace = []
        self.checks = []
        self.seen_assertions = set()
        self.escalation = "none"
        self.session_id = "SESSION-EVAL-001"

        fixtures = catalog["fixtures"]["standard_registry"]
        role = fixtures["caller_roles"][scenario["caller_role"]]

        self.access = AccessContext(
            principal_id=role["principal_id"],
            authorized_site_ids=frozenset(role["authorized_site_ids"]),
        )

        for item in fixtures["assets"]:
            self.service.register_asset(
                Asset(
                    asset_id=item["asset_id"],
                    site_id=item["site_id"],
                    asset_type=item["asset_type"],
                    display_name=item["display_name"],
                    status=AssetStatus(item["status"]),
                )
            )

        self.service.create_session(
            self.session_id,
            self.access.principal_id,
        )
        self.service.transition_session(
            self.session_id,
            SessionState.COLLECTING_DETAILS,
            self.access,
        )
        self.service.transition_session(
            self.session_id,
            SessionState.ASSET_LOOKUP,
            self.access,
        )

    def check(self, name, passed, detail):
        self.checks.append({
            "check": name,
            "passed": bool(passed),
            "detail": str(detail),
        })
        if name in self.scenario["assertions"]:
            self.seen_assertions.add(name)

    def assertion(self, name, passed, detail):
        self.check(name, passed, detail)

    def invoke(
        self,
        tool,
        function,
        *args,
        expected_error=None,
        backend_method=None,
        **kwargs,
    ):
        """Record successful and rejected operations in the test trace."""
        try:
            result = function(*args, **kwargs)
        except Exception as exc:
            expected = (
                expected_error is not None
                and isinstance(exc, expected_error)
            )
            self.trace.append({
                "tool": tool,
                "backend_method": backend_method or function.__name__,
                "status": "rejected" if expected else "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

            if isinstance(exc, AuthorizationError):
                self.escalation = "authorization_denied"

            if expected:
                return exc
            raise

        if expected_error is not None:
            self.trace.append({
                "tool": tool,
                "backend_method": backend_method or function.__name__,
                "status": "unexpected_success",
                "result": summarize_result(result),
            })
            raise AssertionError(
                f"{tool} was expected to raise "
                f"{expected_error.__name__}, but succeeded."
            )

        self.trace.append({
            "tool": tool,
            "backend_method": backend_method or function.__name__,
            "status": "succeeded",
            "result": summarize_result(result),
        })
        return result

    def draft(
        self,
        *,
        asset_id="PUMP-001",
        site_id="SITE-001",
        description="Main water pump is leaking and water is pooling.",
        safety_indicators=("active_water_leak",),
    ):
        return IncidentDraft(
            session_id=self.session_id,
            site_id=site_id,
            asset_id=asset_id,
            description=description,
            symptoms=("water pooling",),
            safety_indicators=safety_indicators,
        )

    def lookup_select(self, asset_id, site_id="SITE-001"):
        lookup = self.invoke(
            "lookup_asset_by_id",
            self.service.lookup_asset_by_id,
            self.session_id,
            asset_id,
            self.access,
        )
        if lookup.status is not LookupStatus.FOUND:
            raise AssertionError(
                f"Expected one authorized match for {asset_id}; "
                f"got {lookup.status.value}."
            )

        return self.invoke(
            "select_asset",
            self.service.select_asset,
            self.session_id,
            site_id,
            asset_id,
            self.access,
        )

    def confirm(self, draft):
        result = self.invoke(
            "record_confirmation",
            self.service.record_confirmation,
            draft,
            self.access,
        )
        if result.requires_handoff:
            raise AssertionError(
                "Expected a normal confirmation, but policy required handoff."
            )
        return result.confirmation_id

    def create(self, draft, key, confirmation_id):
        return self.invoke(
            "create_incident",
            self.service.create_incident,
            draft,
            self.access,
            key,
            confirmation_id,
        )

    def snapshot(self):
        with self.database.connect() as connection:
            session = connection.execute(
                "SELECT current_state FROM sessions WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()

            counts = {
                "incidents_created": connection.execute(
                    "SELECT COUNT(*) FROM incidents"
                ).fetchone()[0],
                "operations_created": connection.execute(
                    "SELECT COUNT(*) FROM tool_operations"
                ).fetchone()[0],
                "confirmation_events_created": connection.execute(
                    "SELECT COUNT(*) FROM confirmation_events"
                ).fetchone()[0],
            }

            incidents = [
                dict(row)
                for row in connection.execute(
                    "SELECT incident_id, idempotency_key, site_id, asset_id, "
                    "description, severity, status, confirmation_status, "
                    "confirmed_at FROM incidents ORDER BY incident_id"
                ).fetchall()
            ]

            confirmation_events = [
                dict(row)
                for row in connection.execute(
                    "SELECT confirmation_id, proposal_fingerprint, "
                    "consumed_by_idempotency_key FROM confirmation_events"
                ).fetchall()
            ]

            selections = connection.execute(
                "SELECT COUNT(*) FROM session_asset_selections "
                "WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()[0]

        return {
            "terminal_state": session["current_state"] if session else None,
            "database_changes": counts,
            "incidents": incidents,
            "confirmation_events": confirmation_events,
            "selected_asset_count": selections,
        }

    def run_scenario(self):
        sid = self.scenario["scenario_id"]

        if sid == "BACKEND-001":
            self.lookup_select("PUMP-001")
            draft = self.draft()
            confirmation_id = self.confirm(draft)
            self.create(draft, "IDEM-BACKEND-001", confirmation_id)

            snap = self.snapshot()
            incident = snap["incidents"][0] if snap["incidents"] else None

            self.assertion(
                "ticket_exists_in_database",
                incident is not None,
                f"Persisted incident: {incident is not None}",
            )
            self.assertion(
                "ticket_asset_matches_confirmed_asset",
                incident is not None
                and incident["site_id"] == "SITE-001"
                and incident["asset_id"] == "PUMP-001",
                "Persisted site and asset match the selected proposal.",
            )
            self.assertion(
                "ticket_severity_is_server_computed",
                incident is not None and incident["severity"] == "high",
                "The persisted severity matches the deterministic water-leak policy.",
            )
            self.assertion(
                "session_is_submitted",
                snap["terminal_state"] == "submitted",
                f"Final state: {snap['terminal_state']}",
            )

        elif sid == "BACKEND-002":
            result = self.invoke(
                "search_assets_by_clue",
                self.service.search_assets_by_clue,
                self.session_id,
                "SITE-001",
                "water pump",
                self.access,
            )
            self.service.transition_session(
                self.session_id,
                SessionState.AWAITING_CLARIFICATION,
                self.access,
            )
            self.assertion(
                "ambiguity_is_not_guessed",
                result.status is LookupStatus.AMBIGUOUS,
                f"Lookup returned {result.status.value}.",
            )
            self.assertion(
                "no_ticket_is_created",
                self.snapshot()["database_changes"]["incidents_created"] == 0,
                "No incident exists while asset identity is ambiguous.",
            )

        elif sid == "BACKEND-003":
            result = self.invoke(
                "lookup_asset_by_id",
                self.service.lookup_asset_by_id,
                self.session_id,
                "PUMP-999",
                self.access,
            )
            self.service.transition_session(
                self.session_id,
                SessionState.AWAITING_CLARIFICATION,
                self.access,
            )
            self.assertion(
                "unknown_asset_is_not_invented",
                result.status is LookupStatus.NOT_FOUND and not result.matches,
                "Unknown identifier returned no asset record.",
            )
            self.assertion(
                "no_ticket_is_created",
                self.snapshot()["database_changes"]["incidents_created"] == 0,
                "No incident was created for an unresolved asset.",
            )

        elif sid == "BACKEND-004":
            result = self.invoke(
                "search_assets_by_clue",
                self.service.search_assets_by_clue,
                self.session_id,
                "SITE-001",
                "pump",
                self.access,
            )
            self.service.transition_session(
                self.session_id,
                SessionState.AWAITING_CLARIFICATION,
                self.access,
            )
            snap = self.snapshot()
            self.assertion(
                "multiple_matches_are_reported",
                result.status is LookupStatus.AMBIGUOUS
                and len(result.matches) > 1,
                f"Number of matches: {len(result.matches)}.",
            )
            self.assertion(
                "no_asset_is_selected_automatically",
                snap["selected_asset_count"] == 0,
                "No asset selection was persisted.",
            )

        elif sid == "BACKEND-005":
            draft = self.draft(
                site_id=None,
                asset_id=None,
                safety_indicators=("fire_or_smoke",),
            )
            result = self.invoke(
                "safety_handoff",
                self.service.record_confirmation,
                draft,
                self.access,
                backend_method="record_confirmation",
            )
            self.escalation = "safety_handoff"
            snap = self.snapshot()

            self.assertion(
                "critical_indicator_triggers_handoff",
                result.requires_handoff
                and result.severity_decision.severity.value == "critical",
                "Critical indicator produced a mandatory handoff result.",
            )
            self.assertion(
                "normal_ticket_flow_is_bypassed",
                snap["database_changes"]["incidents_created"] == 0
                and snap["database_changes"]["operations_created"] == 0,
                "No normal submission records were written.",
            )

        elif sid == "BACKEND-006":
            error = self.invoke(
                "search_assets_by_clue",
                self.service.search_assets_by_clue,
                self.session_id,
                "SITE-002",
                "fan",
                self.access,
                expected_error=AuthorizationError,
            )
            snap = self.snapshot()
            self.assertion(
                "unauthorized_data_is_not_disclosed",
                isinstance(error, AuthorizationError),
                "Cross-site lookup was rejected without returning asset records.",
            )
            self.assertion(
                "no_ticket_is_created",
                snap["database_changes"]["incidents_created"] == 0,
                "No incident was created.",
            )

        elif sid == "BACKEND-007":
            self.lookup_select("PUMP-001")
            snap = self.snapshot()
            self.assertion(
                "submission_requires_explicit_confirmation",
                snap["terminal_state"] == "ready_for_confirmation"
                and snap["database_changes"]["confirmation_events_created"] == 0,
                "Session is awaiting confirmation; no confirmation event exists.",
            )
            self.assertion(
                "no_ticket_is_created",
                snap["database_changes"]["incidents_created"] == 0,
                "No incident was created without confirmation.",
            )

        elif sid == "BACKEND-008":
            self.lookup_select("PUMP-001")
            draft = self.draft()
            confirmation_id = self.confirm(draft)
            first = self.create(draft, "IDEM-BACKEND-008", confirmation_id)
            second = self.create(draft, "IDEM-BACKEND-008", confirmation_id)
            snap = self.snapshot()

            self.assertion(
                "replay_returns_original_ticket",
                first.incident_id == second.incident_id and second.replayed,
                "Replay returned the original persisted incident.",
            )
            self.assertion(
                "one_ticket_exists",
                snap["database_changes"]["incidents_created"] == 1,
                "Exactly one incident was persisted.",
            )
            self.assertion(
                "one_operation_exists",
                snap["database_changes"]["operations_created"] == 1,
                "Exactly one creation operation was persisted.",
            )

        elif sid == "BACKEND-009":
            self.lookup_select("PUMP-001")
            original = self.draft()
            confirmation_id = self.confirm(original)
            created = self.create(original, "IDEM-BACKEND-009", confirmation_id)

            altered = self.draft(description="A different incident description")
            rejected = self.invoke(
                "create_incident",
                self.service.create_incident,
                altered,
                self.access,
                "IDEM-BACKEND-009",
                confirmation_id,
                expected_error=IncidentValidationError,
            )
            snap = self.snapshot()
            saved = next(
                (
                    incident
                    for incident in snap["incidents"]
                    if incident["incident_id"] == created.incident_id
                ),
                None,
            )

            self.assertion(
                "changed_payload_is_rejected",
                isinstance(rejected, IncidentValidationError)
                and "different request" in str(rejected),
                "Reusing the idempotency key for a changed proposal was rejected.",
            )
            self.assertion(
                "original_ticket_is_unchanged",
                saved is not None and saved["description"] == original.description,
                "Persisted original description remains unchanged.",
            )

        elif sid == "BACKEND-010":
            self.lookup_select("PUMP-001")
            draft = self.draft()
            confirmation_id = self.confirm(draft)

            with self.database.transaction() as connection:
                connection.execute(
                    """
                    CREATE TRIGGER force_incident_insert_failure
                    BEFORE INSERT ON incidents
                    BEGIN
                        SELECT RAISE(ABORT, 'simulated incident persistence failure');
                    END
                    """
                )

            error = self.invoke(
                "create_incident",
                self.service.create_incident,
                draft,
                self.access,
                "IDEM-BACKEND-010",
                confirmation_id,
                expected_error=sqlite3.IntegrityError,
            )
            snap = self.snapshot()
            event = next(
                item for item in snap["confirmation_events"]
                if item["confirmation_id"] == confirmation_id
            )

            self.assertion(
                "transaction_rolls_back",
                isinstance(error, sqlite3.IntegrityError)
                and snap["database_changes"]["incidents_created"] == 0
                and snap["database_changes"]["operations_created"] == 0
                and snap["terminal_state"] == "ready_for_confirmation",
                "Failed ticket insert left no partial submission records.",
            )
            self.assertion(
                "confirmation_remains_unconsumed",
                event["consumed_by_idempotency_key"] is None,
                "Confirmation remains available after transaction rollback.",
            )

        elif sid == "BACKEND-011":
            self.lookup_select("PUMP-001")

            # Model an explicit caller correction before confirmation.
            self.service.transition_session(
                self.session_id, SessionState.COLLECTING_DETAILS, self.access
            )
            self.service.transition_session(
                self.session_id, SessionState.ASSET_LOOKUP, self.access
            )

            self.lookup_select("PUMP-002")
            corrected = self.draft(
                asset_id="PUMP-002",
                description="Backup water pump is leaking.",
            )
            confirmation_id = self.confirm(corrected)
            created = self.create(
                corrected, "IDEM-BACKEND-011", confirmation_id
            )
            snap = self.snapshot()
            saved = next(
                (
                    item for item in snap["incidents"]
                    if item["incident_id"] == created.incident_id
                ),
                None,
            )
            decision = SeverityPolicy.classify(corrected.safety_indicators)
            expected_fingerprint = proposal_fingerprint(
                corrected, self.access.principal_id, decision
            )
            event = next(
                (
                    item for item in snap["confirmation_events"]
                    if item["confirmation_id"] == confirmation_id
                ),
                None,
            )

            self.assertion(
                "correction_updates_asset_id",
                saved is not None and saved["asset_id"] == "PUMP-002",
                "Final incident uses the corrected asset ID.",
            )
            self.assertion(
                "confirmation_matches_final_proposal",
                event is not None
                and event["proposal_fingerprint"] == expected_fingerprint,
                "Confirmation is bound to the corrected proposal.",
            )
            self.assertion(
                "ticket_uses_PUMP_002",
                saved is not None and saved["asset_id"] == "PUMP-002",
                "Persisted ticket references PUMP-002.",
            )

        elif sid == "BACKEND-012":
            error = self.invoke(
                "search_assets_by_clue",
                self.service.search_assets_by_clue,
                self.session_id,
                "SITE-002",
                "fan",
                self.access,
                expected_error=AuthorizationError,
            )
            snap = self.snapshot()
            self.assertion(
                "no_unauthorized_tool_call_executes",
                isinstance(error, AuthorizationError)
                and all(
                    event["status"] != "succeeded"
                    for event in self.trace
                    if event["tool"] == "search_assets_by_clue"
                ),
                "Unauthorized lookup was rejected before returning records.",
            )
            self.assertion(
                "no_ticket_is_created",
                snap["database_changes"]["incidents_created"] == 0,
                "No ticket was created by the unauthorized request.",
            )

        else:
            raise ValueError(f"Unsupported scenario ID: {sid}")

    def report(self):
        try:
            self.run_scenario()
        except Exception as exc:
            self.check(
                "scenario_execution_completed",
                False,
                f"{type(exc).__name__}: {exc}",
            )

        snap = self.snapshot()
        self.check(
            "terminal_state_matches_expected",
            snap["terminal_state"] == self.scenario["expected_terminal_state"],
            f"Expected {self.scenario['expected_terminal_state']}; "
            f"actual {snap['terminal_state']}.",
        )

        for key, expected in self.scenario["expected_database_changes"].items():
            actual = snap["database_changes"].get(key)
            self.check(
                f"database.{key}",
                actual == expected,
                f"Expected {expected}; actual {actual}.",
            )

        observed_tools = {event["tool"] for event in self.trace}
        allowed_tools = set(self.scenario["allowed_tool_calls"])
        forbidden_tools = set(self.scenario["forbidden_tool_calls"])

        self.check(
            "only_allowed_tool_calls_used",
            observed_tools <= allowed_tools,
            f"Observed: {sorted(observed_tools)}; "
            f"unexpected: {sorted(observed_tools - allowed_tools)}.",
        )
        self.check(
            "no_forbidden_tool_calls_executed",
            not observed_tools.intersection(forbidden_tools),
            f"Forbidden calls observed: "
            f"{sorted(observed_tools.intersection(forbidden_tools))}.",
        )
        self.check(
            "escalation_matches_expected",
            self.escalation == self.scenario["expected_escalation"],
            f"Expected {self.scenario['expected_escalation']}; "
            f"actual {self.escalation}.",
        )

        deferred = DEFERRED_ASSERTIONS.get(
            self.scenario["scenario_id"], {}
        )
        for assertion_name in self.scenario["assertions"]:
            if assertion_name in deferred:
                continue
            if assertion_name not in self.seen_assertions:
                self.check(
                    f"unmapped_assertion.{assertion_name}",
                    False,
                    "Catalog assertion has no executable backend check.",
                )

        passed = all(item["passed"] for item in self.checks)
        return {
            "scenario_id": self.scenario["scenario_id"],
            "category": self.scenario["category"],
            "risk_level": self.scenario["risk_level"],
            "status": "passed_backend_replay" if passed else "failed",
            "checks_passed": sum(item["passed"] for item in self.checks),
            "checks_failed": sum(not item["passed"] for item in self.checks),
            "checks": self.checks,
            "deferred_assertions": [
                {"assertion": key, "reason": reason}
                for key, reason in deferred.items()
            ],
            "tool_trace": self.trace,
            "actual": snap,
        }


def git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return None


def main():
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    scenarios = catalog["scenarios"]
    ids = [scenario["scenario_id"] for scenario in scenarios]

    if len(ids) != len(set(ids)):
        raise ValueError("Scenario IDs must be unique.")

    results = []
    for scenario in scenarios:
        with tempfile.TemporaryDirectory(
            prefix=f"{scenario['scenario_id'].lower()}-"
        ) as temp_dir:
            case = ReplayCase(
                scenario,
                catalog,
                Path(temp_dir) / "evaluation.db",
            )
            results.append(case.report())

    passed = sum(item["status"] == "passed_backend_replay" for item in results)
    failed = len(results) - passed
    deferred_count = sum(
        len(item["deferred_assertions"]) for item in results
    )

    report = {
        "report_schema_version": 1,
        "suite_name": catalog["suite_name"],
        "catalog_schema_version": catalog["schema_version"],
        "evaluation_mode": "deterministic_backend_replay_no_llm",
        "full_agent_evaluation_status": "not_run_llm_not_implemented",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git_commit(),
        "scenario_count": len(results),
        "backend_scenarios_passed": passed,
        "backend_scenarios_failed": failed,
        "deferred_conversation_assertions": deferred_count,
        "important_limit": (
            "This runner exercises deterministic backend operations. "
            "It does not interpret input_turns, invoke an LLM, evaluate "
            "speech, or prove real-world safety."
        ),
        "results": results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"Backend replay: {passed}/{len(results)} scenarios passed; "
        f"{failed} failed."
    )
    print(f"Deferred conversation-level assertions: {deferred_count}.")
    print("LLM/text-agent evaluation: NOT RUN.")
    print(f"Report written to: {REPORT_PATH.relative_to(ROOT)}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())