"""Deterministic incident severity policy.

Initial prototype rules. Requires domain-expert validation before real use.
"""

from dataclasses import dataclass

from field_incident_agent.models import Severity


CRITICAL_INDICATORS = frozenset({
    "fire_or_smoke",
    "gas_leak",
    "electrical_hazard",
    "structural_instability",
    "injury_or_immediate_danger",
})

HIGH_INDICATORS = frozenset({
    "active_water_leak",
    "essential_service_disrupted",
    "equipment_unusable",
})

MEDIUM_INDICATORS = frozenset({
    "equipment_degraded",
    "repeated_fault",
})

LOW_INDICATORS = frozenset({
    "cosmetic_only",
})

KNOWN_INDICATORS = (
    CRITICAL_INDICATORS
    | HIGH_INDICATORS
    | MEDIUM_INDICATORS
    | LOW_INDICATORS
)


@dataclass(frozen=True, slots=True)
class SeverityDecision:
    severity: Severity
    requires_handoff: bool
    reason: str
    indicators: tuple[str, ...]


class SeverityPolicy:
    """Map approved indicator codes to severity using fixed precedence."""

    @staticmethod
    def classify(indicators: tuple[str, ...]) -> SeverityDecision:
        if not isinstance(indicators, tuple):
            raise ValueError("Indicators must be a tuple of approved codes.")

        normalized_values = []

        for indicator in indicators:
            if not isinstance(indicator, str):
                raise ValueError("Every indicator must be a string.")

            normalized_indicator = indicator.strip().casefold()

            if not normalized_indicator:
                raise ValueError("Indicators cannot be blank.")

            normalized_values.append(normalized_indicator)

        # Validate types before normalization and deduplication.
        normalized = tuple(dict.fromkeys(normalized_values))

        unknown = set(normalized) - KNOWN_INDICATORS
        if unknown:
            raise ValueError(
                f"Unsupported safety indicators: {sorted(unknown)}"
            )

        if CRITICAL_INDICATORS.intersection(normalized):
            return SeverityDecision(
                severity=Severity.CRITICAL,
                requires_handoff=True,
                reason="critical_safety_indicator",
                indicators=normalized,
            )

        if HIGH_INDICATORS.intersection(normalized):
            return SeverityDecision(
                severity=Severity.HIGH,
                requires_handoff=False,
                reason="high_impact_indicator",
                indicators=normalized,
            )

        if MEDIUM_INDICATORS.intersection(normalized):
            return SeverityDecision(
                severity=Severity.MEDIUM,
                requires_handoff=False,
                reason="moderate_impact_indicator",
                indicators=normalized,
            )

        if LOW_INDICATORS.intersection(normalized):
            return SeverityDecision(
                severity=Severity.LOW,
                requires_handoff=False,
                reason="explicit_cosmetic_only",
                indicators=normalized,
            )

        return SeverityDecision(
            severity=Severity.MEDIUM,
            requires_handoff=False,
            reason="no_approved_indicator_conservative_default",
            indicators=normalized,
        )