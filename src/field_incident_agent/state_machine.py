"""Deterministic session-state transition rules."""

from field_incident_agent.models import SessionState


_ALLOWED_TRANSITIONS: dict[SessionState, frozenset[SessionState]] = {
    SessionState.STARTED: frozenset({
        SessionState.COLLECTING_DETAILS,
        SessionState.HANDOFF_REQUIRED,
        SessionState.FAILED_FINAL,
        SessionState.CANCELLED,
    }),
    SessionState.COLLECTING_DETAILS: frozenset({
        SessionState.ASSET_LOOKUP,
        SessionState.AWAITING_CLARIFICATION,
        SessionState.HANDOFF_REQUIRED,
        SessionState.FAILED_RECOVERABLE,
        SessionState.FAILED_FINAL,
        SessionState.CANCELLED,
    }),
    SessionState.ASSET_LOOKUP: frozenset({
        SessionState.AWAITING_CLARIFICATION,
        SessionState.READY_FOR_CONFIRMATION,
        SessionState.HANDOFF_REQUIRED,
        SessionState.FAILED_RECOVERABLE,
        SessionState.FAILED_FINAL,
        SessionState.CANCELLED,
    }),
    SessionState.AWAITING_CLARIFICATION: frozenset({
        SessionState.COLLECTING_DETAILS,
        SessionState.ASSET_LOOKUP,
        SessionState.READY_FOR_CONFIRMATION,
        SessionState.HANDOFF_REQUIRED,
        SessionState.FAILED_RECOVERABLE,
        SessionState.FAILED_FINAL,
        SessionState.CANCELLED,
    }),
    SessionState.READY_FOR_CONFIRMATION: frozenset({
        SessionState.COLLECTING_DETAILS,
        SessionState.SUBMITTING,
        SessionState.HANDOFF_REQUIRED,
        SessionState.FAILED_FINAL,
        SessionState.CANCELLED,
    }),
    SessionState.SUBMITTING: frozenset({
        SessionState.SUBMITTED,
        SessionState.FAILED_RECOVERABLE,
        SessionState.FAILED_FINAL,
        SessionState.HANDOFF_REQUIRED,
    }),
    SessionState.FAILED_RECOVERABLE: frozenset({
        SessionState.COLLECTING_DETAILS,
        SessionState.ASSET_LOOKUP,
        SessionState.AWAITING_CLARIFICATION,
        SessionState.READY_FOR_CONFIRMATION,
        SessionState.SUBMITTING,
        SessionState.HANDOFF_REQUIRED,
        SessionState.FAILED_FINAL,
        SessionState.CANCELLED,
    }),
    SessionState.SUBMITTED: frozenset(),
    SessionState.HANDOFF_REQUIRED: frozenset(),
    SessionState.FAILED_FINAL: frozenset(),
    SessionState.CANCELLED: frozenset(),
}


def transition_session_state(
    current: SessionState,
    target: SessionState,
    *,
    retry_safe: bool = False,
) -> SessionState:
    """Validate a state change and return the target state.

    retry_safe must only be true after operation-status reconciliation
    confirms that retrying the same intended mutation is safe.
    """
    if not isinstance(current, SessionState):
        raise ValueError("current must be a SessionState.")

    if not isinstance(target, SessionState):
        raise ValueError("target must be a SessionState.")

    if current == target:
        raise ValueError("Self-transitions are not allowed.")

    if target not in _ALLOWED_TRANSITIONS[current]:
        raise ValueError(
            f"Transition from '{current.value}' to '{target.value}' "
            "is not allowed."
        )

    if (
        current is SessionState.FAILED_RECOVERABLE
        and target is SessionState.SUBMITTING
        and retry_safe is not True
    ):
        raise ValueError(
            "A submission retry requires confirmed operation status "
            "and authorization to retry safely."
        )

    return target