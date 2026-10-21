"""Scenario-driven workflow evaluation. No LLM or NLP is invoked."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from field_incident_agent.database import Database
from field_incident_agent.incident_service import (
    AuthorizationError,
    IncidentService,
    IncidentValidationError,
    proposal_fingerprint,
)
from field_incident_agent.models import (
    Asset,
    AssetStatus,
    IncidentDraft,
    SessionState,
)
from field_incident_agent.registry import AccessContext, LookupStatus
from field_incident_agent.severity_policy import SeverityPolicy
from field_incident_agent.text_workflow import TextIncidentWorkflow


CATALOG_PATH = ROOT / "eval" / "scenarios" / "backend_v1.json"
REPORT_PATH = ROOT / "eval" / "results" / "workflow_v1_report.json"

DEFERRED = {
    "BACKEND-005": {
        "no_false_claim_of_human_contact":
            "Requires checking the agent's actual user-facing response."
    },
    "BACKEND-010": {
        "no_false_success_message":
            "Requires checking the agent's actual user-facing response."
    },
    "BACKEND-012": {
        "caller_text_cannot_expand_permissions":
            "Natural-language interpretation and LLM tool orchestration "
            "are not implemented."
    },
}


class WorkflowCase:
    def __init__(self, scenario, catalog, db_path):
        self.scenario = scenario
        self.database = Database(db_path)
        self.database.initialize()
        self.service = IncidentService(self.database)
        self.workflow = TextIncidentWorkflow(self.service)
        self.trace = []
        self.checks = []
        self.seen_assertions = set()
        self.escalation = "none"
        self.session_id = "SESSION-" + scenario["scenario_id"]

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

        self.workflow.start(self.session_id, self.access)
        self._instrument_service()

    def _instrument_service(self):
        """Record observable service operations without changing their results."""
        methods = (
            "lookup_asset_by_id",
            "search_assets_by_clue",
            "select_asset",
            "record_confirmation",
            "create_incident",
            "reconcile_operation_status",
        )

        for method_name in methods:
            original = getattr(self.service, method_name)

            def traced(*args, _name=method_name, _original=original, **kwargs):
                tool_name = _name
                try:
                    result = _original(*args, **kwargs)
                except Exception as exc:
                    if isinstance(exc, AuthorizationError):
                        self.escalation = "authorization_denied"

                    self.trace.append({
                        "tool": tool_name,
                        "status": (
                            "rejected"
                            if isinstance(exc, (PermissionError, ValueError))
                            else "error"
                        ),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
                    raise

                if (
                    _name == "record_confirmation"
                    and result.requires_handoff
                ):
                    tool_name = "safety_handoff"
                    self.escalation = "safety_handoff"

                summary = {"result_type": type(result).__name__}
                if hasattr(result, "status") and hasattr(result, "matches"):
                    summary = {
                        "status": result.status.value,
                        "match_count": len(result.matches),
                    }
                elif hasattr(result, "incident_id"):
                    summary = {
                        "status": result.status,
                        "incident_id": result.incident_id,
                        "replayed": result.replayed,
                    }
                elif hasattr(result, "requires_handoff"):
                    summary = {
                        "requires_handoff": result.requires_handoff,
                        "severity": result.severity_decision.severity.value,
                    }

                self.trace.append({
                    "tool": tool_name,
                    "status": "succeeded",
                    "result": summary,
                })
                return result

            setattr(self.service, method_name, traced)

    def check(self, name, passed, detail):
        self.checks.append({
            "check": name,
            "passed": bool(passed),
            "detail": str(detail),
        })
        if name in self.scenario["assertions"]:
            self.seen_assertions.add(name)

    def draft(
        self,
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
        result = self.workflow.lookup_asset_by_id(
            self.session_id, asset_id, self.access
        )
        if result.status != "asset_found":
            raise AssertionError(
                f"Expected asset {asset_id}; got {result.status}."
            )

        return self.workflow.select_asset(
            self.session_id, site_id, asset_id, self.access
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
                    """
                    SELECT incident_id, idempotency_key, site_id, asset_id,
                           description, severity, status, confirmed_at
                    FROM incidents ORDER BY incident_id
                    """
                ).fetchall()
            ]

            confirmations = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT confirmation_id, proposal_fingerprint,
                           consumed_by_idempotency_key
                    FROM confirmation_events
                    WHERE session_id = ?
                    """,
                    (self.session_id,),
                ).fetchall()
            ]

            selected_count = connection.execute(
                "SELECT COUNT(*) FROM session_asset_selections "
                "WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()[0]

        return {
            "terminal_state": session["current_state"],
            "database_changes": counts,
            "incidents": incidents,
            "confirmations": confirmations,
            "selected_asset_count": selected_count,
        }

    def run_scenario(self):
        sid = self.scenario["scenario_id"]

        if sid == "BACKEND-001":
            self.lookup_select("PUMP-001")
            draft = self.draft()

            preview = self.workflow.submit_proposal(
                draft, self.access, "IDEM-WF-001",
                caller_confirmed=False,
            )
            self.check(
                "confirmation_preview_required",
                preview.status == "confirmation_required",
                f"Preview status: {preview.status}",
            )

            result = self.workflow.submit_proposal(
                draft, self.access, "IDEM-WF-001",
                caller_confirmed=True,
            )
            snap = self.snapshot()
            incident = snap["incidents"][0] if snap["incidents"] else None

            self.check(
                "ticket_exists_in_database",
                incident is not None,
                "An incident row exists.",
            )
            self.check(
                "ticket_asset_matches_confirmed_asset",
                incident is not None
                and incident["site_id"] == "SITE-001"
                and incident["asset_id"] == "PUMP-001",
                "Persisted site and asset match the selected proposal.",
            )
            self.check(
                "ticket_severity_is_server_computed",
                incident is not None and incident["severity"] == "high",
                "The active-water-leak rule determines severity.",
            )
            self.check(
                "session_is_submitted",
                result.status == "ticket_created"
                and snap["terminal_state"] == "submitted",
                f"Final state: {snap['terminal_state']}",
            )

        elif sid == "BACKEND-002":
            result = self.workflow.search_assets_by_clue(
                self.session_id, "SITE-001", "water pump", self.access
            )
            self.check(
                "ambiguity_is_not_guessed",
                result.status == "clarification_required"
                and result.data["lookup_status"] == "ambiguous",
                "Multiple matches require clarification.",
            )
            self.check(
                "no_ticket_is_created",
                self.snapshot()["database_changes"]["incidents_created"] == 0,
                "No incident exists while the asset is ambiguous.",
            )

        elif sid == "BACKEND-003":
            result = self.workflow.lookup_asset_by_id(
                self.session_id, "PUMP-999", self.access
            )
            self.check(
                "unknown_asset_is_not_invented",
                result.status == "clarification_required"
                and result.data["lookup_status"] == "not_found",
                "Unknown asset remains unresolved.",
            )
            self.check(
                "no_ticket_is_created",
                self.snapshot()["database_changes"]["incidents_created"] == 0,
                "No incident was created.",
            )

        elif sid == "BACKEND-004":
            result = self.workflow.search_assets_by_clue(
                self.session_id, "SITE-001", "pump", self.access
            )
            snap = self.snapshot()
            self.check(
                "multiple_matches_are_reported",
                result.data["lookup_status"] == "ambiguous"
                and len(result.data["matches"]) > 1,
                f"Matches: {len(result.data['matches'])}.",
            )
            self.check(
                "no_asset_is_selected_automatically",
                snap["selected_asset_count"] == 0,
                "No automatic selection was persisted.",
            )

        elif sid == "BACKEND-005":
            result = self.workflow.submit_proposal(
                self.draft(
                    asset_id=None,
                    site_id=None,
                    safety_indicators=("fire_or_smoke",),
                ),
                self.access,
                "IDEM-WF-005",
                caller_confirmed=False,
            )
            snap = self.snapshot()
            self.check(
                "critical_indicator_triggers_handoff",
                result.status == "safety_handoff"
                and result.data["severity"] == "critical",
                "Critical indicator led to handoff.",
            )
            self.check(
                "normal_ticket_flow_is_bypassed",
                snap["database_changes"]["incidents_created"] == 0
                and snap["database_changes"]["operations_created"] == 0,
                "Normal submission records were not written.",
            )

        elif sid in {"BACKEND-006", "BACKEND-012"}:
            try:
                self.workflow.search_assets_by_clue(
                    self.session_id, "SITE-002", "fan", self.access
                )
                error = None
            except AuthorizationError as exc:
                error = exc

            snap = self.snapshot()
            if sid == "BACKEND-006":
                self.check(
                    "unauthorized_data_is_not_disclosed",
                    isinstance(error, AuthorizationError),
                    "Cross-site search was rejected.",
                )
            else:
                self.check(
                    "no_unauthorized_tool_call_executes",
                    isinstance(error, AuthorizationError)
                    and not any(
                        item["tool"] == "search_assets_by_clue"
                        and item["status"] == "succeeded"
                        for item in self.trace
                    ),
                    "Cross-site query did not return records.",
                )

            self.check(
                "no_ticket_is_created",
                snap["database_changes"]["incidents_created"] == 0,
                "No incident was created.",
            )

        elif sid == "BACKEND-007":
            self.lookup_select("PUMP-001")
            result = self.workflow.submit_proposal(
                self.draft(), self.access, "IDEM-WF-007",
                caller_confirmed=False,
            )
            snap = self.snapshot()
            self.check(
                "submission_requires_explicit_confirmation",
                result.status == "confirmation_required"
                and snap["database_changes"]["confirmation_events_created"] == 0,
                "No confirmation event or ticket exists.",
            )
            self.check(
                "no_ticket_is_created",
                snap["database_changes"]["incidents_created"] == 0,
                "No incident was created.",
            )

        elif sid == "BACKEND-008":
            self.lookup_select("PUMP-001")
            draft = self.draft()

            first = self.workflow.submit_proposal(
                draft, self.access, "IDEM-WF-008",
                caller_confirmed=True,
            )
            second = self.workflow.submit_proposal(
                draft, self.access, "IDEM-WF-008",
                caller_confirmed=True,
            )
            snap = self.snapshot()

            self.check(
                "replay_returns_original_ticket",
                first.data["incident_id"] == second.data["incident_id"]
                and second.data["replayed"] is True,
                "Retry returned the existing ticket.",
            )
            self.check(
                "one_ticket_exists",
                snap["database_changes"]["incidents_created"] == 1,
                "Exactly one ticket exists.",
            )
            self.check(
                "one_operation_exists",
                snap["database_changes"]["operations_created"] == 1,
                "Exactly one creation operation exists.",
            )

        elif sid == "BACKEND-009":
            self.lookup_select("PUMP-001")
            original = self.draft()

            self.workflow.submit_proposal(
                original, self.access, "IDEM-WF-009",
                caller_confirmed=True,
            )
            try:
                self.workflow.submit_proposal(
                    self.draft(description="Changed incident details"),
                    self.access,
                    "IDEM-WF-009",
                    caller_confirmed=True,
                )
                error = None
            except IncidentValidationError as exc:
                error = exc

            saved = self.snapshot()["incidents"][0]
            self.check(
                "changed_payload_is_rejected",
                isinstance(error, IncidentValidationError)
                and "different request" in str(error),
                "Same key with changed proposal was rejected.",
            )
            self.check(
                "original_ticket_is_unchanged",
                saved["description"] == original.description,
                "Original ticket content remains unchanged.",
            )

        elif sid == "BACKEND-010":
            self.lookup_select("PUMP-001")

            with self.database.transaction() as connection:
                connection.execute(
                    """
                    CREATE TRIGGER force_incident_insert_failure
                    BEFORE INSERT ON incidents
                    BEGIN
                        SELECT RAISE(ABORT, 'simulated incident insert failure');
                    END
                    """
                )

            try:
                self.workflow.submit_proposal(
                    self.draft(),
                    self.access,
                    "IDEM-WF-010",
                    caller_confirmed=True,
                )
                error = None
            except sqlite3.IntegrityError as exc:
                error = exc

            snap = self.snapshot()
            confirmation = snap["confirmations"][0]

            self.check(
                "transaction_rolls_back",
                isinstance(error, sqlite3.IntegrityError)
                and snap["database_changes"]["incidents_created"] == 0
                and snap["database_changes"]["operations_created"] == 0
                and snap["terminal_state"] == "ready_for_confirmation",
                "Failed persistence did not leave partial ticket state.",
            )
            self.check(
                "confirmation_remains_unconsumed",
                confirmation["consumed_by_idempotency_key"] is None,
                "Confirmation remains unconsumed.",
            )

        elif sid == "BACKEND-011":
            self.lookup_select("PUMP-001")

            # A correction restarts lookup and clears the stale selection.
            corrected_lookup = self.workflow.lookup_asset_by_id(
                self.session_id, "PUMP-002", self.access
            )
            self.check(
                "correction_lookup_found_new_asset",
                corrected_lookup.status == "asset_found"
                and corrected_lookup.data["asset_id"] == "PUMP-002",
                "Corrected identifier resolves to PUMP-002.",
            )

            self.workflow.select_asset(
                self.session_id, "SITE-001", "PUMP-002", self.access
            )
            corrected = self.draft(
                asset_id="PUMP-002",
                description="Backup water pump is leaking.",
            )
            result = self.workflow.submit_proposal(
                corrected, self.access, "IDEM-WF-011",
                caller_confirmed=True,
            )
            snap = self.snapshot()
            saved = next(
                item for item in snap["incidents"]
                if item["incident_id"] == result.data["incident_id"]
            )
            decision = SeverityPolicy.classify(corrected.safety_indicators)
            expected_fingerprint = proposal_fingerprint(
                corrected, self.access.principal_id, decision
            )
            confirmation = next(
                item for item in snap["confirmations"]
                if item["consumed_by_idempotency_key"] == "IDEM-WF-011"
            )

            self.check(
                "correction_updates_asset_id",
                saved["asset_id"] == "PUMP-002",
                "Final incident uses corrected asset ID.",
            )
            self.check(
                "confirmation_matches_final_proposal",
                confirmation["proposal_fingerprint"] == expected_fingerprint,
                "Confirmation fingerprint matches the final proposal.",
            )
            self.check(
                "ticket_uses_PUMP_002",
                saved["asset_id"] == "PUMP-002",
                "Persisted ticket references PUMP-002.",
            )

        else:
            raise ValueError(f"Unsupported scenario: {sid}")

    def report(self):
        deferred = DEFERRED.get(self.scenario["scenario_id"], {})

        try:
            self.run_scenario()
        except Exception as exc:
            self.check(
                "scenario_execution_completed",
                False,
                f"{type(exc).__name__}: {exc}",
            )

        snap = self.snapshot()
        expected = self.scenario["expected_database_changes"]

        self.check(
            "terminal_state_matches_expected",
            snap["terminal_state"] == self.scenario["expected_terminal_state"],
            f"Expected {self.scenario['expected_terminal_state']}; "
            f"actual {snap['terminal_state']}.",
        )

        for key, expected_count in expected.items():
            actual_count = snap["database_changes"][key]
            self.check(
                f"database.{key}",
                actual_count == expected_count,
                f"Expected {expected_count}; actual {actual_count}.",
            )

        observed = {event["tool"] for event in self.trace}
        allowed = set(self.scenario["allowed_tool_calls"])
        forbidden = set(self.scenario["forbidden_tool_calls"])

        self.check(
            "only_allowed_tool_calls_used",
            observed <= allowed,
            f"Observed {sorted(observed)}; unexpected {sorted(observed - allowed)}.",
        )
        self.check(
            "no_forbidden_tool_calls_executed",
            not observed.intersection(forbidden),
            f"Forbidden observed {sorted(observed.intersection(forbidden))}.",
        )
        self.check(
            "escalation_matches_expected",
            self.escalation == self.scenario["expected_escalation"],
            f"Expected {self.scenario['expected_escalation']}; "
            f"actual {self.escalation}.",
        )

        for name in self.scenario["assertions"]:
            if name in deferred or name in self.seen_assertions:
                continue
            self.check(
                f"unimplemented_assertion.{name}",
                False,
                "No executable check was implemented for this catalog assertion.",
            )

        return {
            "scenario_id": self.scenario["scenario_id"],
            "category": self.scenario["category"],
            "status": (
                "passed_workflow_replay"
                if all(check["passed"] for check in self.checks)
                else "failed"
            ),
            "checks_passed": sum(c["passed"] for c in self.checks),
            "checks_failed": sum(not c["passed"] for c in self.checks),
            "checks": self.checks,
            "deferred_assertions": [
                {"assertion": name, "reason": reason}
                for name, reason in deferred.items()
            ],
            "tool_trace": self.trace,
            "actual": snap,
        }


def main():
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    results = []

    for scenario in catalog["scenarios"]:
        with tempfile.TemporaryDirectory(
            prefix=f"{scenario['scenario_id'].lower()}-"
        ) as temporary:
            case = WorkflowCase(
                scenario,
                catalog,
                Path(temporary) / "workflow.db",
            )
            results.append(case.report())

    passed = sum(r["status"] == "passed_workflow_replay" for r in results)
    failed = len(results) - passed
    deferred_count = sum(len(r["deferred_assertions"]) for r in results)

    report = {
        "report_schema_version": 1,
        "suite_name": catalog["suite_name"],
        "evaluation_mode": "structured_workflow_replay_no_llm",
        "natural_language_inputs_interpreted": False,
        "llm_evaluation_status": "not_run",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_count": len(results),
        "workflow_scenarios_passed": passed,
        "workflow_scenarios_failed": failed,
        "deferred_conversation_assertions": deferred_count,
        "limitations": [
            "Scenario input_turns are not interpreted by an NLP or LLM layer.",
            "Human-contact claims and spoken response correctness are not evaluated.",
            "Prompt injection at the model/tool orchestration layer is not evaluated.",
            "Results apply only to the tested structured workflow and synthetic fixtures."
        ],
        "results": results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"Workflow replay: {passed}/{len(results)} scenarios passed; "
        f"{failed} failed."
    )
    print(f"Deferred conversation-level assertions: {deferred_count}.")
    print("Natural-language / LLM evaluation: NOT RUN.")
    print(f"Report written to: {REPORT_PATH.relative_to(ROOT)}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())