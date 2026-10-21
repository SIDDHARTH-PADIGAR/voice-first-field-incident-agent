import pytest

from field_incident_agent.models import SessionState
from field_incident_agent.state_machine import transition_session_state


def test_valid_workflow_progression():
    assert transition_session_state(
        SessionState.STARTED,
        SessionState.COLLECTING_DETAILS,
    ) is SessionState.COLLECTING_DETAILS

    assert transition_session_state(
        SessionState.COLLECTING_DETAILS,
        SessionState.ASSET_LOOKUP,
    ) is SessionState.ASSET_LOOKUP

    assert transition_session_state(
        SessionState.ASSET_LOOKUP,
        SessionState.READY_FOR_CONFIRMATION,
    ) is SessionState.READY_FOR_CONFIRMATION

    assert transition_session_state(
        SessionState.READY_FOR_CONFIRMATION,
        SessionState.SUBMITTING,
    ) is SessionState.SUBMITTING

    assert transition_session_state(
        SessionState.SUBMITTING,
        SessionState.SUBMITTED,
    ) is SessionState.SUBMITTED


def test_caller_correction_returns_to_detail_collection():
    assert transition_session_state(
        SessionState.READY_FOR_CONFIRMATION,
        SessionState.COLLECTING_DETAILS,
    ) is SessionState.COLLECTING_DETAILS


def test_handoff_can_happen_before_submission():
    assert transition_session_state(
        SessionState.COLLECTING_DETAILS,
        SessionState.HANDOFF_REQUIRED,
    ) is SessionState.HANDOFF_REQUIRED


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (SessionState.STARTED, SessionState.SUBMITTING),
        (SessionState.COLLECTING_DETAILS, SessionState.SUBMITTED),
        (SessionState.ASSET_LOOKUP, SessionState.SUBMITTING),
        (SessionState.SUBMITTING, SessionState.CANCELLED),
    ],
)
def test_invalid_transitions_are_rejected(current, target):
    with pytest.raises(ValueError, match="not allowed"):
        transition_session_state(current, target)


def test_self_transition_is_rejected():
    with pytest.raises(ValueError, match="Self-transitions"):
        transition_session_state(
            SessionState.COLLECTING_DETAILS,
            SessionState.COLLECTING_DETAILS,
        )


@pytest.mark.parametrize(
    "terminal",
    [
        SessionState.SUBMITTED,
        SessionState.HANDOFF_REQUIRED,
        SessionState.FAILED_FINAL,
        SessionState.CANCELLED,
    ],
)
def test_terminal_states_cannot_transition(terminal):
    with pytest.raises(ValueError, match="not allowed"):
        transition_session_state(
            terminal,
            SessionState.COLLECTING_DETAILS,
        )


def test_submission_retry_requires_reconciliation():
    with pytest.raises(ValueError, match="operation status"):
        transition_session_state(
            SessionState.FAILED_RECOVERABLE,
            SessionState.SUBMITTING,
        )


def test_reconciled_retry_can_enter_submitting():
    assert transition_session_state(
        SessionState.FAILED_RECOVERABLE,
        SessionState.SUBMITTING,
        retry_safe=True,
    ) is SessionState.SUBMITTING


def test_raw_strings_are_not_accepted_as_states():
    with pytest.raises(ValueError, match="current"):
        transition_session_state(
            "started",
            SessionState.COLLECTING_DETAILS,
        )