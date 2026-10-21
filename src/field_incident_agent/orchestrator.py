"""Connect untrusted model extraction to the controlled text workflow."""

from dataclasses import dataclass

from field_incident_agent.incident_service import (
    IncidentService,
    IncidentValidationError,
)
from field_incident_agent.llm_client import (
    IncidentExtractionClient,
    ModelClientError,
)
from field_incident_agent.llm_contract import (
    IncidentExtraction,
    parse_incident_extraction,
)
from field_incident_agent.models import IncidentDraft, SessionState
from field_incident_agent.registry import AccessContext, LookupStatus
from field_incident_agent.severity_policy import SeverityPolicy
from field_incident_agent.text_workflow import (
    TextIncidentWorkflow,
    WorkflowReply,
)


@dataclass(frozen=True, slots=True)
class IntakeOutcome:
    """Internal result. Keep the draft server-side until confirmation."""

    reply: WorkflowReply
    draft: IncidentDraft | None = None
    extraction: IncidentExtraction | None = None


class IncidentOrchestrator:
    """Turn one caller utterance into a safe, structured workflow proposal."""

    def __init__(
        self,
        client: IncidentExtractionClient,
        workflow: TextIncidentWorkflow,
        service: IncidentService,
    ) -> None:
        self.client = client
        self.workflow = workflow
        self.service = service

    def process_report(
        self,
        session_id: str,
        caller_text: str,
        access: AccessContext,
        idempotency_key: str,
    ) -> IntakeOutcome:
        """Extract and validate one utterance, then resolve the asset.

        This is a single-turn adapter, not a general-purpose conversation
        manager. Confirmation is a separate trusted application action.
        """
        if not isinstance(caller_text, str) or not caller_text.strip():
            return IntakeOutcome(
                WorkflowReply(
                    "clarification_required",
                    "Please describe the equipment problem.",
                    {},
                )
            )

        if len(caller_text) > 5000:
            return IntakeOutcome(
                WorkflowReply(
                    "input_rejected",
                    "The report is too long. Please summarize the issue.",
                    {},
                )
            )

        try:
            raw_output = self.client.extract_incident(caller_text)
        except ModelClientError:
            return IntakeOutcome(
                WorkflowReply(
                    "model_unavailable",
                    "I could not interpret the report. Please retry or hand it off.",
                    {},
                )
            )

        try:
            extraction = parse_incident_extraction(raw_output)
        except (ValueError, TypeError) as exc:
            # Do not pass malformed output to the workflow or expose it as
            # an authoritative incident.
            return IntakeOutcome(
                WorkflowReply(
                    "invalid_extraction",
                    "The report could not be safely structured. Please clarify it.",
                    {"reason": str(exc)},
                )
            )

        decision = SeverityPolicy.classify(extraction.safety_indicators)

        description = extraction.description or (
            "Safety concern reported; details remain unresolved."
            if decision.requires_handoff
            else None
        )

        if decision.requires_handoff:
            draft = IncidentDraft(
                session_id=session_id,
                site_id=extraction.site_id,
                asset_id=extraction.asset_id,
                asset_lookup_clue=extraction.asset_lookup_clue,
                description=description,
                symptoms=extraction.symptoms,
                safety_indicators=extraction.safety_indicators,
                time_noticed=extraction.time_noticed,
                operational_impact=extraction.operational_impact,
            )
            reply = self.workflow.submit_proposal(
                draft,
                access,
                idempotency_key,
                caller_confirmed=False,
            )
            return IntakeOutcome(reply, draft, extraction)

        if extraction.safety_concern_uncertain:
            return IntakeOutcome(
                WorkflowReply(
                    "safety_clarification_required",
                    "I need to clarify the potential safety concern before continuing.",
                    {"incident_created": False},
                ),
                extraction=extraction,
            )

        if description is None:
            return IntakeOutcome(
                WorkflowReply(
                    "clarification_required",
                    "Please describe what happened to the equipment.",
                    {},
                ),
                extraction=extraction,
            )

        # Reject a stated site that is outside the authenticated caller's
        # permissions before making a lookup.
        if (
            extraction.site_id is not None
            and extraction.site_id not in access.authorized_site_ids
        ):
            return IntakeOutcome(
                WorkflowReply(
                    "authorization_denied",
                    "That site is not available to your account.",
                    {},
                ),
                extraction=extraction,
            )

        if extraction.asset_id is None and extraction.asset_lookup_clue is None:
            return IntakeOutcome(
                WorkflowReply(
                    "clarification_required",
                    "Please provide an equipment ID or a description we can search.",
                    {},
                ),
                extraction=extraction,
            )

        if extraction.asset_id is not None:
            lookup = self.workflow.lookup_asset_by_id(
                session_id,
                extraction.asset_id,
                access,
            )
        else:
            if extraction.site_id is None:
                return IntakeOutcome(
                    WorkflowReply(
                        "clarification_required",
                        "Which site is this equipment located at?",
                        {},
                    ),
                    extraction=extraction,
                )

            lookup = self.workflow.search_assets_by_clue(
                session_id,
                extraction.site_id,
                extraction.asset_lookup_clue,
                access,
            )

        if lookup.status != "asset_found":
            return IntakeOutcome(lookup, extraction=extraction)

        asset_id = lookup.data["asset_id"]
        site_id = lookup.data["site_id"]

        if extraction.site_id is not None and extraction.site_id != site_id:
            state = self.service.get_session_state(session_id, access)
            if state is SessionState.ASSET_LOOKUP:
                self.service.transition_session(
                    session_id,
                    SessionState.AWAITING_CLARIFICATION,
                    access,
                )

            return IntakeOutcome(
                WorkflowReply(
                    "clarification_required",
                    "The equipment and site details do not agree. Please verify them.",
                    {},
                ),
                extraction=extraction,
            )

        self.workflow.select_asset(
            session_id,
            site_id,
            asset_id,
            access,
        )

        draft = IncidentDraft(
            session_id=session_id,
            site_id=site_id,
            asset_id=asset_id,
            asset_lookup_clue=extraction.asset_lookup_clue,
            description=description,
            symptoms=extraction.symptoms,
            safety_indicators=extraction.safety_indicators,
            time_noticed=extraction.time_noticed,
            operational_impact=extraction.operational_impact,
        )

        reply = self.workflow.submit_proposal(
            draft,
            access,
            idempotency_key,
            caller_confirmed=False,
        )
        return IntakeOutcome(reply, draft, extraction)

    def confirm_proposal(
        self,
        outcome: IntakeOutcome,
        access: AccessContext,
        idempotency_key: str,
        *,
        explicitly_confirmed: bool,
    ) -> WorkflowReply:
        """Submit only the server-held proposal that the caller reviewed."""
        if not isinstance(explicitly_confirmed, bool):
            raise TypeError("explicitly_confirmed must be a boolean.")

        if (
            outcome.draft is None
            or outcome.reply.status != "confirmation_required"
        ):
            raise IncidentValidationError(
                "There is no normal incident proposal awaiting confirmation."
            )

        if not explicitly_confirmed:
            return outcome.reply

        return self.workflow.submit_proposal(
            outcome.draft,
            access,
            idempotency_key,
            caller_confirmed=True,
        )