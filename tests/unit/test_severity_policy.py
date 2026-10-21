import pytest

from field_incident_agent.models import Severity
from field_incident_agent.severity_policy import SeverityPolicy


@pytest.mark.parametrize(
    "indicator",
    [
        "fire_or_smoke",
        "gas_leak",
        "electrical_hazard",
        "structural_instability",
        "injury_or_immediate_danger",
    ],
)
def test_critical_indicators_require_handoff(indicator):
    decision = SeverityPolicy.classify((indicator,))

    assert decision.severity is Severity.CRITICAL
    assert decision.requires_handoff is True


@pytest.mark.parametrize(
    "indicator",
    [
        "active_water_leak",
        "essential_service_disrupted",
        "equipment_unusable",
    ],
)
def test_high_impact_indicators_produce_high_severity(indicator):
    decision = SeverityPolicy.classify((indicator,))

    assert decision.severity is Severity.HIGH
    assert decision.requires_handoff is False


def test_moderate_impact_produces_medium_severity():
    decision = SeverityPolicy.classify(("equipment_degraded",))

    assert decision.severity is Severity.MEDIUM


def test_explicit_cosmetic_issue_produces_low_severity():
    decision = SeverityPolicy.classify(("cosmetic_only",))

    assert decision.severity is Severity.LOW


def test_missing_indicators_default_to_medium():
    decision = SeverityPolicy.classify(())

    assert decision.severity is Severity.MEDIUM
    assert decision.reason == (
        "no_approved_indicator_conservative_default"
    )


def test_critical_indicator_overrides_low_indicator():
    decision = SeverityPolicy.classify(
        ("cosmetic_only", "fire_or_smoke")
    )

    assert decision.severity is Severity.CRITICAL
    assert decision.requires_handoff is True


def test_unknown_indicator_is_rejected():
    with pytest.raises(ValueError, match="Unsupported safety indicators"):
        SeverityPolicy.classify(("probably_safe",))


def test_string_instead_of_tuple_is_rejected():
    with pytest.raises(ValueError, match="must be a tuple"):
        SeverityPolicy.classify("fire_or_smoke")


def test_indicator_whitespace_and_duplicates_are_normalized():
    decision = SeverityPolicy.classify(
        (" equipment_degraded ", "equipment_degraded")
    )

    assert decision.severity is Severity.MEDIUM
    assert decision.indicators == ("equipment_degraded",)