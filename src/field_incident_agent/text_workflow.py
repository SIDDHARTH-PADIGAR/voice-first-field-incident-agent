"""Deterministic text-workflow orchestration; no LLM or NLP yet."""

from dataclasses import dataclass

from field_incident_agent.incident_service import (
    IncidentService,
    IncidentValidationError,
)
from field_incident_agent.models import (
    IncidentDraft,
    SessionState,
)
from field_incident_agent.registry import AccessContext, LookupStatus
from field_incident_agent.severity_policy import SeverityPolicy


@dataclass(frozen=True, slots=True)
class WorkflowReply:
    status: str
    message: str
    data: dict[str, object]


class TextIncidentWorkflow:
    """Orchestrate intake operations using explicit, already-extracted fields."""

    def __init__(self, service: IncidentService) -> None:
        self.service = service

    def start(
        self,
        session_id: str,
        access: AccessContext,
    ) -> WorkflowReply:
        self.service.create_session(session_id, access.principal_id)
        self.service.transition_session(
            session_id, SessionState.COLLECTING_DETAILS, access
        )
        return WorkflowReply(
            status="collecting_details",
            message="Describe the incident and identify the site or equipment.",
            data={"session_id": session_id},
        )

    def _prepare_lookup(
        self,
        session_id: str,
        access: AccessContext,
    ) -> None:
        state = self.service.get_session_state(session_id, access)

        if state is SessionState.ASSET_LOOKUP:
            return

        if state in {
            SessionState.STARTED,
            SessionState.COLLECTING_DETAILS,
            SessionState.AWAITING_CLARIFICATION,
            SessionState.READY_FOR_CONFIRMATION,
        }:
            if state is not SessionState.COLLECTING_DETAILS:
                self.service.transition_session(
                    session_id, SessionState.COLLECTING_DETAILS, access
                )

            self.service.transition_session(
                session_id, SessionState.ASSET_LOOKUP, access
            )
            return

        if state is SessionState.FAILED_RECOVERABLE:
            self.service.transition_session(
                session_id, SessionState.ASSET_LOOKUP, access
            )
            return

        raise IncidentValidationError(
            f"Asset lookup is not allowed from state '{state.value}'."
        )

    def lookup_asset_by_id(
        self,
        session_id: str,
        asset_id: str,
        access: AccessContext,
    ) -> WorkflowReply:
        self._prepare_lookup(session_id, access)
        result = self.service.lookup_asset_by_id(
            session_id, asset_id, access
        )

        if result.status is not LookupStatus.FOUND:
            self.service.transition_session(
                session_id, SessionState.AWAITING_CLARIFICATION, access
            )
            return WorkflowReply(
                status="clarification_required",
                message=(
                    "The asset was not uniquely identified. "
                    "Verify its identifier or choose from the returned matches."
                ),
                data={
                    "lookup_status": result.status.value,
                    "matches": [
                        {
                            "site_id": asset.site_id,
                            "asset_id": asset.asset_id,
                            "display_name": asset.display_name,
                        }
                        for asset in result.matches
                    ],
                },
            )

        asset = result.matches[0]
        return WorkflowReply(
            status="asset_found",
            message="Asset found. Confirm that this is the equipment involved.",
            data={
                "site_id": asset.site_id,
                "asset_id": asset.asset_id,
                "display_name": asset.display_name,
            },
        )

    def search_assets_by_clue(
        self,
        session_id: str,
        site_id: str,
        clue: str,
        access: AccessContext,
    ) -> WorkflowReply:
        self._prepare_lookup(session_id, access)
        result = self.service.search_assets_by_clue(
            session_id, site_id, clue, access
        )

        if result.status is not LookupStatus.FOUND:
            self.service.transition_session(
                session_id, SessionState.AWAITING_CLARIFICATION, access
            )
            return WorkflowReply(
                status="clarification_required",
                message=(
                    "The asset wasn't uniquely identified. "
                    "Please clarify which equipment is involved."
                ),
                data={
                    "lookup_status": result.status.value,
                    "matches": [
                        {
                            "site_id": asset.site_id,
                            "asset_id": asset.asset_id,
                            "display_name": asset.display_name,
                        }
                        for asset in result.matches
                    ],
                },
            )

        asset = result.matches[0]
        return WorkflowReply(
            status="asset_found",
            message="Asset found. Confirm that this is the equipment involved.",
            data={
                "site_id": asset.site_id,
                "asset_id": asset.asset_id,
                "display_name": asset.display_name,
            },
        )

    def select_asset(
        self,
        session_id: str,
        site_id: str,
        asset_id: str,
        access: AccessContext,
    ) -> WorkflowReply:
        asset = self.service.select_asset(
            session_id, site_id, asset_id, access
        )
        return WorkflowReply(
            status="ready_for_confirmation",
            message=(
                "Asset selected. Review the proposed incident and confirm "
                "before it is submitted."
            ),
            data={
                "site_id": asset.site_id,
                "asset_id": asset.asset_id,
                "display_name": asset.display_name,
            },
        )

    def _creation_reply(self, result) -> WorkflowReply:
        succeeded = result.status == "succeeded"
        return WorkflowReply(
            status="ticket_created" if succeeded else result.status,
            message=(
                "Incident persisted successfully."
                if succeeded
                else "Incident outcome is not confirmed; check operation status."
            ),
            data={
                "incident_id": result.incident_id,
                "operation_id": result.operation_id,
                "severity": result.severity.value,
                "replayed": result.replayed,
            },
        )

    def submit_proposal(
        self,
        draft: IncidentDraft,
        access: AccessContext,
        idempotency_key: str,
        *,
        caller_confirmed: bool,
    ) -> WorkflowReply:
        """Preview, confirm, and submit; resolve retries before new writes."""
        if not isinstance(caller_confirmed, bool):
            raise TypeError("caller_confirmed must be a boolean.")

        state = self.service.get_session_state(draft.session_id, access)
        decision = SeverityPolicy.classify(draft.safety_indicators)

        if state is SessionState.HANDOFF_REQUIRED:
            return WorkflowReply(
                status="safety_handoff",
                message="This incident requires a safety handoff.",
                data={"severity": decision.severity.value},
            )

        # Existing operation lookup happens before new proposal validation.
        # It can replay the original result or reject a key/payload conflict.
        if state in {
            SessionState.SUBMITTED,
            SessionState.SUBMITTING,
            SessionState.FAILED_RECOVERABLE,
        }:
            result = self.service.create_incident(
                draft, access, idempotency_key, None
            )
            return self._creation_reply(result)

        if decision.requires_handoff:
            result = self.service.record_confirmation(draft, access)
            return WorkflowReply(
                status="safety_handoff",
                message=(
                    "A critical safety indicator requires handoff. "
                    "Normal incident submission was not performed."
                ),
                data={
                    "severity": result.severity_decision.severity.value,
                    "reason": result.severity_decision.reason,
                    "incident_created": False,
                },
            )

        decision = self.service.validate_proposal(draft, access)
        proposal = {
            "site_id": draft.site_id,
            "asset_id": draft.asset_id,
            "description": draft.description,
            "symptoms": list(draft.symptoms),
            "severity": decision.severity.value,
            "severity_reason": decision.reason,
        }

        if not caller_confirmed:
            return WorkflowReply(
                status="confirmation_required",
                message=(
                    "Review the incident details and explicitly confirm "
                    "submission."
                ),
                data={"proposal": proposal},
            )

        confirmation = self.service.record_confirmation(draft, access)

        if confirmation.requires_handoff:
            return WorkflowReply(
                status="safety_handoff",
                message="This incident requires a safety handoff.",
                data={"incident_created": False},
            )

        result = self.service.create_incident(
            draft,
            access,
            idempotency_key,
            confirmation.confirmation_id,
        )
        return self._creation_reply(result)