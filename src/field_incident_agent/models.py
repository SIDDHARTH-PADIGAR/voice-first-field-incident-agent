"""Validated domain models for field incident intake."""

from dataclasses import dataclass
from enum import StrEnum
import re


_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")


class AssetStatus(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(StrEnum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


class SessionState(StrEnum):
    STARTED = "started"
    COLLECTING_DETAILS = "collecting_details"
    ASSET_LOOKUP = "asset_lookup"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    READY_FOR_CONFIRMATION = "ready_for_confirmation"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    HANDOFF_REQUIRED = "handoff_required"
    FAILED_RECOVERABLE = "failed_recoverable"
    FAILED_FINAL = "failed_final"
    CANCELLED = "cancelled"


def _validate_id(value: str, field: str) -> None:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field} must be 1-64 characters using letters, digits, "
            "periods, underscores, colons, or hyphens; it cannot "
            "start with punctuation."
        )


def _validate_text(
    value: str,
    field: str,
    max_length: int,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string.")

    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{field} cannot be empty.")

    if len(normalized) > max_length:
        raise ValueError(
            f"{field} cannot exceed {max_length} characters."
        )

    return normalized


def _validate_optional_text(
    value: str | None,
    field: str,
    max_length: int,
) -> str | None:
    if value is None:
        return None

    return _validate_text(value, field, max_length)


def _validate_text_tuple(
    values: tuple[str, ...],
    field: str,
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{field} must be a tuple of strings.")

    if len(values) > 20:
        raise ValueError(f"{field} cannot contain more than 20 items.")

    return tuple(
        _validate_text(value, f"{field} item", 250)
        for value in values
    )


@dataclass(frozen=True, slots=True)
class Asset:
    asset_id: str
    site_id: str
    asset_type: str
    display_name: str
    status: AssetStatus = AssetStatus.ACTIVE

    def __post_init__(self) -> None:
        _validate_id(self.asset_id, "asset_id")
        _validate_id(self.site_id, "site_id")

        object.__setattr__(
            self,
            "asset_type",
            _validate_text(self.asset_type, "asset_type", 80),
        )
        object.__setattr__(
            self,
            "display_name",
            _validate_text(self.display_name, "display_name", 120),
        )

        if not isinstance(self.status, AssetStatus):
            raise ValueError("status must be an AssetStatus value.")


@dataclass(frozen=True, slots=True)
class IncidentDraft:
    """Extracted report; unresolved fields remain explicitly unknown."""

    session_id: str
    description: str
    site_id: str | None = None
    asset_id: str | None = None
    asset_lookup_clue: str | None = None
    symptoms: tuple[str, ...] = ()
    safety_indicators: tuple[str, ...] = ()
    time_noticed: str | None = None
    operational_impact: str | None = None

    def __post_init__(self) -> None:
        _validate_id(self.session_id, "session_id")

        for field in ("site_id", "asset_id"):
            value = getattr(self, field)
            if value is not None:
                _validate_id(value, field)

        for field, limit in (
            ("description", 2000),
            ("asset_lookup_clue", 300),
            ("time_noticed", 120),
            ("operational_impact", 500),
        ):
            value = getattr(self, field)
            if field == "description" or value is not None:
                object.__setattr__(
                    self,
                    field,
                    _validate_text(value, field, limit),
                )

        object.__setattr__(
            self,
            "symptoms",
            _validate_text_tuple(self.symptoms, "symptoms"),
        )
        object.__setattr__(
            self,
            "safety_indicators",
            _validate_text_tuple(
                self.safety_indicators,
                "safety_indicators",
            ),
        )