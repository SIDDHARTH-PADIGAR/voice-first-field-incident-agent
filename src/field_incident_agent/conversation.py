"""Stateful conversational turns over the controlled incident workflow."""

from dataclasses import asdict, dataclass
import re
import secrets
import uuid

from field_incident_agent.incident_service import IncidentValidationError
from field_incident_agent.orchestrator import IntakeOutcome
from field_incident_agent.registry import AccessContext
from field_incident_agent.runtime import IncidentRuntime


DEMO_SITE_ID = "SITE-DEMO"
DEMO_PRINCIPAL_ID = "TECH-DEMO"
MAX_CONVERSATIONS = 100

def _normalize_reply(text: str) -> str:
    """Normalize short caller replies for deterministic intent matching."""
    return re.sub(
        r"\s+",
        " ",
        re.sub(r"[^a-z0-9\s]", "", text.casefold()),
    ).strip()


_AFFIRMATIVE = {
    "yes",
    "yes create it",
    "yes create the ticket",
    "yes submit it",
    "create the ticket",
    "confirm",
    "confirm it",
    "go ahead",
    "submit it",
    "thats correct",
    "yes thats correct",
    "looks good",
    "yeah create it",
    "yep create it",
}

_DECLINE = {
    "no",
    "no thanks",
    "no dont create it",
    "dont create it",
    "do not create it",
    "no dont submit it",
    "dont submit it",
    "not yet",
}

_CANCEL = {"cancel", "cancel it", "discard", "discard it"}
_RETRY = {"retry", "try again", "please retry"}

_NEW_INCIDENT = {
    "new incident",
    "start a new incident",
    "start new incident",
    "report a new incident",
    "another incident",
    "start another incident",
    "new report",
    "start a new report",
    "different incident",
}

_TERMINAL_ACKS = (
    _AFFIRMATIVE
    | _DECLINE
    | _CANCEL
    | _RETRY
    | {
        "ok", "okay", "thanks", "thank you", "understood",
        "got it", "nope", "nah", "why", "what",
        "what do you mean", "not sure", "i dont know",
    }
)


def _looks_like_new_incident(text: str) -> bool:
    """Recognize a clear new report without treating acknowledgements as one."""
    normalized = _normalize_reply(text)

    if normalized in _TERMINAL_ACKS:
        return False
    if normalized in _NEW_INCIDENT:
        return True

    starts_new_report = re.match(
        r"^(?:at site\b|another issue\b|different issue\b|new incident\b|new report\b)",
        normalized,
    )
    has_incident_details = re.search(
        r"\b(?:DEMO-[A-Z0-9-]+|pump|elevator|lift|panel|leak|leaking|"
        r"smoke|smoking|flames|fire|sparks|vibrat\w*|stopped|broken|"
        r"fault|water|equipment|machine|outage)\b",
        text,
        re.IGNORECASE,
    )
    return bool(starts_new_report and has_incident_details)

@dataclass(slots=True)
class _Conversation:
    conversation_id: str
    report: str
    revision_count: int = 0
    last_status: str = "new"
    outcome: IntakeOutcome | None = None
    idempotency_key: str | None = None
    proposal_id: str | None = None
    extraction: dict[str, object] | None = None
    workflow_data: dict[str, object] | None = None


class ConversationManager:
    """Manage one active incident proposal per conversation."""

    def __init__(self, runtime: IncidentRuntime) -> None:
        self.runtime = runtime
        self._conversations: dict[str, _Conversation] = {}
        self._proposal_to_conversation: dict[str, str] = {}

    @staticmethod
    def _access() -> AccessContext:
        # Explicitly synthetic demo identity. Not an authentication mechanism.
        return AccessContext(
            principal_id=DEMO_PRINCIPAL_ID,
            authorized_site_ids=frozenset({DEMO_SITE_ID}),
        )

    def _new_conversation(self, report: str) -> _Conversation:
        state = _Conversation(
            conversation_id=f"CONV-WEB-{uuid.uuid4().hex}",
            report=report,
        )
        self._conversations[state.conversation_id] = state

        while len(self._conversations) > MAX_CONVERSATIONS:
            oldest_id = next(iter(self._conversations))
            oldest = self._conversations.pop(oldest_id)
            if oldest.proposal_id:
                self._proposal_to_conversation.pop(oldest.proposal_id, None)

        return state

    def _invalidate_proposal(self, state: _Conversation) -> None:
        if state.proposal_id:
            self._proposal_to_conversation.pop(state.proposal_id, None)
        state.proposal_id = None

    def has_proposal(self, proposal_id: str) -> bool:
        return proposal_id in self._proposal_to_conversation

    def _asset_display_name(self, asset_id: str | None) -> str | None:
        if not asset_id:
            return None

        with self.runtime.database.connect() as connection:
            row = connection.execute(
                "SELECT display_name FROM assets WHERE asset_id = ?",
                (asset_id,),
            ).fetchone()

        return row["display_name"] if row else None

    def _agent_message(
        self,
        state: _Conversation,
        status: str,
        data: dict[str, object],
        outcome: IntakeOutcome | None,
    ) -> str:
        if status == "confirmation_required":
            proposal = data.get("proposal", {})
            if not isinstance(proposal, dict):
                proposal = {}

            asset_id = proposal.get("asset_id")
            site_id = proposal.get("site_id")
            asset_name = self._asset_display_name(
                asset_id if isinstance(asset_id, str) else None
            )

            if asset_name:
                asset_sentence = f"I found the {asset_name}"
            elif asset_id:
                asset_sentence = f"I found asset {asset_id}"
            else:
                asset_sentence = "I haven't identified the exact equipment yet"

            if site_id:
                asset_sentence += f" at site {site_id}"

            description = str(proposal.get("description") or "the reported issue")
            description = description.rstrip(" .")
            severity = str(proposal.get("severity") or "undetermined")

            prefix = (
                "Thanks for clarifying. Here's the updated understanding. "
                if state.revision_count
                else "Here's what I understood. "
            )

            return (
                f"{prefix}{asset_sentence}. {description}. "
                f"Under the current demo rules, this is classified as "
                f"{severity} severity. Did I get that right? Say "
                f"\"yes, create the ticket,\" or tell me what needs correcting."
            )

        if status == "safety_handoff":
            return (
                "This report describes a critical safety concern. I've stopped "
                "normal ticket creation and marked the intake for a safety "
                "handoff. This demo has not notified a person, and no normal "
                "incident ticket was created."
            )

        if status == "safety_clarification_required":
            return (
                "I'm not certain about a possible safety concern. Please tell "
                "me more about the condition so we can resolve that uncertainty. "
                "I haven't created a ticket."
            )

        if status == "clarification_required":
            matches = data.get("matches", [])
            if isinstance(matches, list) and matches:
                options = []
                for match in matches:
                    if isinstance(match, dict):
                        name = match.get("display_name", "Possible asset")
                        asset_id = match.get("asset_id")
                        label = str(name)
                        if asset_id:
                            label += f" (asset ID {asset_id})"
                        options.append(label)

                return (
                    "I found more than one possible asset: "
                    + "; ".join(options)
                    + ". Which one is involved?"
                )

            lookup_status = data.get("lookup_status")
            if lookup_status == "not_found":
                return (
                    "I couldn't find an authorized asset matching that description. "
                    "Could you give me its exact asset ID or clarify which site "
                    "it's at?"
                )

            if outcome and outcome.extraction:
                return (
                    "I need one more detail before I can proceed. "
                    + outcome.reply.message
                )

            return (
                "I need a little more information before I can continue. "
                "Which site or equipment are you reporting?"
            )

        if status == "model_unavailable":
            return (
                "I couldn't interpret that report because the model is unavailable. "
                "Nothing was submitted. You can say \"retry\" or try again."
            )

        if status == "invalid_extraction":
            return (
                "I couldn't safely structure that report. Nothing was submitted. "
                "Please rephrase the incident and try again."
            )

        if status == "authorization_denied":
            return (
                "I can't proceed with that site because it's not available to "
                "this demo account. No ticket was created."
            )

        if status == "input_rejected":
            return (
                "That report is too long for this demo. Please summarize the issue "
                "in a shorter message."
            )

        if status == "awaiting_correction":
            return (
                "Okay, I won't create that ticket. What should I correct? "
                "Tell me the detail that was wrong."
            )

        if status == "cancelled":
            return (
                "Okay, I've discarded that proposal. No ticket was created. "
                "You can describe another incident whenever you're ready."
            )

        if status == "ticket_created":
            incident_id = data.get("incident_id")
            if incident_id:
                return (
                    f"The incident ticket has been saved in the local demo database. "
                    f"Its reference is {incident_id}. No external maintenance "
                    f"system was contacted."
                )
            return (
                "The incident ticket has been saved in the local demo database. "
                "No external maintenance system was contacted."
            )

        return (
            outcome.reply.message
            if outcome
            else "I couldn't complete that step. No success is being claimed."
        )

    def _response(
        self,
        state: _Conversation,
        status: str,
        message: str,
        agent_message: str,
        data: dict[str, object] | None = None,
    ) -> dict[str, object]:
        response_data = data if data is not None else (
            state.workflow_data or {}
        )

        proposal = (
            response_data.get("proposal")
            if status == "confirmation_required"
            else None
        )
        proposal_id = (
            state.proposal_id if status == "confirmation_required" else None
        )

        return {
            "conversation_id": state.conversation_id,
            "status": status,
            "message": message,
            "agent_message": agent_message,
            "data": response_data,
            "extraction": state.extraction,
            "proposal": proposal,
            "proposal_id": proposal_id,
            "revision": state.revision_count,
        }

    def _analyze(self, state: _Conversation) -> dict[str, object]:
        if len(state.report) > 5000:
            raise ValueError(
                "This conversation has become too long. Start a new incident "
                "with a shorter report."
            )

        self._invalidate_proposal(state)
        access = self._access()
        session_id = f"SESSION-WEB-{uuid.uuid4().hex}"
        idempotency_key = f"IDEM-WEB-{uuid.uuid4().hex}"

        self.runtime.workflow.start(session_id, access)
        outcome = self.runtime.orchestrator.process_report(
            session_id,
            state.report,
            access,
            idempotency_key,
        )

        state.outcome = outcome
        state.idempotency_key = idempotency_key
        state.last_status = outcome.reply.status
        state.workflow_data = outcome.reply.data
        state.extraction = (
            asdict(outcome.extraction)
            if outcome.extraction is not None
            else None
        )

        if state.last_status == "confirmation_required":
            state.proposal_id = secrets.token_urlsafe(32)
            self._proposal_to_conversation[state.proposal_id] = (
                state.conversation_id
            )

        return self._response(
            state,
            state.last_status,
            outcome.reply.message,
            self._agent_message(
                state,
                state.last_status,
                outcome.reply.data,
                outcome,
            ),
            outcome.reply.data,
        )

    def _confirm(self, state: _Conversation) -> dict[str, object]:
        if (
            state.outcome is None
            or state.idempotency_key is None
            or state.proposal_id is None
        ):
            return self._response(
                state,
                "clarification_required",
                "There is no pending proposal to confirm.",
                "I don't have an incident proposal awaiting confirmation. "
                "Tell me what happened first.",
                {},
            )

        reply = self.runtime.orchestrator.confirm_proposal(
            state.outcome,
            self._access(),
            state.idempotency_key,
            explicitly_confirmed=True,
        )
        state.last_status = reply.status
        state.workflow_data = reply.data

        agent_message = self._agent_message(
            state,
            reply.status,
            reply.data,
            None,
        )
        return self._response(
            state,
            reply.status,
            reply.message,
            agent_message,
            reply.data,
        )

    def confirm_by_proposal(self, proposal_id: str) -> dict[str, object]:
        conversation_id = self._proposal_to_conversation.get(proposal_id)
        if conversation_id is None:
            raise KeyError("Proposal not found.")

        state = self._conversations.get(conversation_id)
        if (
            state is None
            or state.proposal_id != proposal_id
            or state.outcome is None
        ):
            raise KeyError("Proposal not found.")

        if state.last_status not in {"confirmation_required", "ticket_created"}:
            raise KeyError("Proposal is no longer awaiting confirmation.")

        return self._confirm(state)

    def turn(
        self,
        conversation_id: str | None,
        text: str,
    ) -> dict[str, object]:
        """Handle reports, corrections, clarification, and confirmation."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Please enter or say a message.")

        text = text.strip()
        if len(text) > 5000:
            raise ValueError("A single message cannot exceed 5000 characters.")

        normalized = _normalize_reply(text)
        state = (
            self._conversations.get(conversation_id)
            if conversation_id
            else None
        )

        # Never silently leave a critical safety handoff because of a yes/no
        # reply. A new incident must be explicitly requested.
        if state and state.last_status == "safety_handoff":
            if normalized in _NEW_INCIDENT:
                state = self._new_conversation("")
                state.last_status = "awaiting_new_incident"
                message = "Okay. Describe the separate incident, and I'll start a new intake."
                return self._response(
                    state, state.last_status, message, message, {}
                )

            if normalized in _AFFIRMATIVE:
                message = (
                    "I can't create a normal ticket from this report because "
                    "it remains flagged for critical safety handoff. This demo "
                    "has not notified a person. Say 'start a new incident' to "
                    "report a separate issue."
                )
            elif normalized in _DECLINE or normalized in _CANCEL:
                message = (
                    "The report remains flagged for safety handoff, and this "
                    "demo has not notified a person. When you said 'No', what "
                    "were you disagreeing with? I can't clear the safety handoff "
                    "from a yes/no reply. Say 'start a new incident' for a "
                    "separate report."
                )
            else:
                message = (
                    "I still have this report flagged for safety handoff. Are "
                    "you clarifying the existing report, or do you want to "
                    "report a separate incident? I won't clear the handoff "
                    "automatically. Say 'start a new incident' to begin a "
                    "separate report."
                )

            return self._response(
                state,
                "safety_handoff",
                message,
                message,
                state.workflow_data or {},
            )

        # A completed ticket remains completed. Short replies must not start
        # another intake or create a duplicate ticket.
        if state and state.last_status == "ticket_created":
            if normalized in _NEW_INCIDENT:
                state = self._new_conversation("")
                state.last_status = "awaiting_new_incident"
                message = "Okay. Describe the separate incident."
                return self._response(
                    state, state.last_status, message, message, {}
                )

            if not _looks_like_new_incident(text):
                incident_id = (state.workflow_data or {}).get("incident_id")
                reference = (
                    f" Its local demo reference is {incident_id}."
                    if incident_id
                    else ""
                )
                message = (
                    "That ticket has already been created, so I won't create "
                    "another one from this reply."
                    + reference
                    + " If you're reporting a different maintenance issue, "
                    "describe it or say 'start a new incident'."
                )
                return self._response(
                    state,
                    "ticket_created",
                    message,
                    message,
                    state.workflow_data or {},
                )

            # A clear, substantive new report can begin after a completed ticket.
            state = None

        if state and state.last_status == "cancelled":
            if normalized in _NEW_INCIDENT:
                state = self._new_conversation("")
                state.last_status = "awaiting_new_incident"
                message = "Okay. Describe the new incident."
                return self._response(
                    state, state.last_status, message, message, {}
                )

            if not _looks_like_new_incident(text):
                message = (
                    "This intake was cancelled and no ticket was created. "
                    "Describe a new incident or say 'start a new incident' "
                    "when you're ready."
                )
                return self._response(
                    state, "cancelled", message, message, {}
                )
            state = None

        if state is None:
            state = self._new_conversation(text)

            if normalized in _AFFIRMATIVE:
                state.last_status = "clarification_required"
                message = (
                    "I don't have an incident proposal awaiting confirmation. "
                    "Tell me what happened first."
                )
                return self._response(
                    state, state.last_status, message, message, {}
                )

            return self._analyze(state)

        if state.last_status == "awaiting_new_incident":
            state.report = text
            state.revision_count = 0
            return self._analyze(state)

        if (
            state.last_status == "confirmation_required"
            and state.outcome is not None
        ):
            if normalized in _AFFIRMATIVE:
                return self._confirm(state)

            if normalized in _CANCEL:
                self._invalidate_proposal(state)
                state.outcome = None
                state.last_status = "cancelled"
                message = "The pending proposal was discarded. No ticket was created."
                return self._response(
                    state,
                    "cancelled",
                    message,
                    self._agent_message(state, "cancelled", {}, None),
                    {},
                )

            if normalized in _DECLINE:
                self._invalidate_proposal(state)
                state.outcome = None
                state.last_status = "awaiting_correction"
                message = "The proposal was declined. No ticket was created. What should I correct?"
                return self._response(
                    state,
                    "awaiting_correction",
                    message,
                    self._agent_message(state, "awaiting_correction", {}, None),
                    {},
                )

            addition = (
                "\n\nCALLER CORRECTION (newer explicit details override "
                "conflicting earlier details): " + text
            )
            candidate = state.report + addition
            if len(candidate) > 5000:
                raise ValueError(
                    "That correction makes the report too long. "
                    "Please restate the incident and correction more briefly."
                )

            state.report = candidate
            state.revision_count += 1
            return self._analyze(state)

        if (
            state.last_status == "model_unavailable"
            and normalized in _RETRY
        ):
            return self._analyze(state)

        if normalized in _AFFIRMATIVE:
            message = (
                "I don't have a proposal awaiting confirmation. "
                "Please describe the incident first."
            )
            return self._response(
                state, "clarification_required", message, message, {}
            )

        addition = (
            "\n\nCALLER FOLLOW-UP (newer information takes precedence "
            "where it conflicts with earlier details): " + text
        )
        candidate = state.report + addition
        if len(candidate) > 5000:
            raise ValueError(
                "That follow-up makes the report too long. Start a new "
                "incident with a shorter report."
            )

        state.report = candidate
        state.revision_count += 1
        return self._analyze(state)

