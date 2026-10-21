"""Provider-neutral interface for incident extraction clients."""

from collections.abc import Mapping
from typing import Protocol, TypeAlias


ModelOutput: TypeAlias = str | Mapping[str, object]


class ModelClientError(RuntimeError):
    """The model provider failed to return a usable response."""


class IncidentExtractionClient(Protocol):
    """Extract incident fields; never authorize or execute operations."""

    def extract_incident(self, text: str) -> ModelOutput:
        """Return a JSON string or mapping matching the extraction contract."""
        ...