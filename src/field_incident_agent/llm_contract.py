"""Strict contract for untrusted LLM incident extraction output."""

from collections.abc import Mapping
from dataclasses import dataclass
import json
import re

from field_incident_agent.severity_policy import KNOWN_INDICATORS


_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")

_REQUIRED_FIELDS = {
    "site_id",
    "asset_id",
    "asset_lookup_clue",
    "description",
    "symptoms",
    "safety_indicators",
    "safety_concern_uncertain",
    "time_noticed",
    "operational_impact",
}


@dataclass(frozen=True, slots=True)
class IncidentExtraction:
    """Untrusted, validated extraction. Not an authorized action."""

    site_id: str | None
    asset_id: str | None
    asset_lookup_clue: str | None
    description: str | None
    symptoms: tuple[str, ...]
    safety_indicators: tuple[str, ...]
    safety_concern_uncertain: bool
    time_noticed: str | None
    operational_impact: str | None


def _optional_text(value: object, field: str, limit: int) -> str | None:
    if value is None:
        return None

    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null.")

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must be null or non-empty text.")

    if len(normalized) > limit:
        raise ValueError(f"{field} cannot exceed {limit} characters.")

    return normalized


def _optional_id(value: object, field: str) -> str | None:
    normalized = _optional_text(value, field, 64)

    if normalized is not None and not _ID_PATTERN.fullmatch(normalized):
        raise ValueError(f"{field} is not a valid identifier.")

    return normalized


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array.")

    if len(value) > 20:
        raise ValueError(f"{field} cannot contain more than 20 items.")

    results = []
    for item in value:
        normalized = _optional_text(item, f"{field} item", 250)
        if normalized is None:
            raise ValueError(f"{field} items cannot be null.")
        results.append(normalized)

    return tuple(dict.fromkeys(results))


def parse_incident_extraction(
    response: str | Mapping[str, object],
) -> IncidentExtraction:
    """Parse model output and reject malformed or unexpected fields.

    Never accept a severity, permission, tool name, confirmation status,
    ticket ID, or workflow state from the model.
    """
    if isinstance(response, str):
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError("Model output must be valid JSON.") from exc
    else:
        payload = response

    if not isinstance(payload, Mapping):
        raise ValueError("Model output must be a JSON object.")

    missing = _REQUIRED_FIELDS - set(payload)
    unexpected = set(payload) - _REQUIRED_FIELDS

    if missing:
        raise ValueError(f"Missing extraction fields: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected extraction fields: {sorted(unexpected)}")

    site_id = _optional_id(payload["site_id"], "site_id")
    asset_id = _optional_id(payload["asset_id"], "asset_id")
    clue = _optional_text(payload["asset_lookup_clue"], "asset_lookup_clue", 300)
    description = _optional_text(payload["description"], "description", 2000)
    symptoms = _string_tuple(payload["symptoms"], "symptoms")
    indicators = _string_tuple(
        payload["safety_indicators"], "safety_indicators"
    )

    normalized_indicators = tuple(
        dict.fromkeys(item.strip().casefold() for item in indicators)
    )
    unknown = set(normalized_indicators) - KNOWN_INDICATORS
    if unknown:
        raise ValueError(
            f"Unsupported safety indicators: {sorted(unknown)}"
        )

    uncertain = payload["safety_concern_uncertain"]
    if not isinstance(uncertain, bool):
        raise ValueError("safety_concern_uncertain must be a boolean.")

    return IncidentExtraction(
        site_id=site_id,
        asset_id=asset_id,
        asset_lookup_clue=clue,
        description=description,
        symptoms=symptoms,
        safety_indicators=normalized_indicators,
        safety_concern_uncertain=uncertain,
        time_noticed=_optional_text(
            payload["time_noticed"], "time_noticed", 120
        ),
        operational_impact=_optional_text(
            payload["operational_impact"], "operational_impact", 500
        ),
    )