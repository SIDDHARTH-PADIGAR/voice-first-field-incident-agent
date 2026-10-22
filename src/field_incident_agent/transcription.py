from __future__ import annotations

import os
from typing import Protocol


class TranscriptionError(RuntimeError):
    """A transcription request failed or returned an invalid response."""


class TranscriptionConfigurationError(TranscriptionError):
    """The transcription provider is not configured correctly."""


class AudioTranscriber(Protocol):
    def transcribe(
        self,
        audio: bytes,
        *,
        filename: str,
        content_type: str,
    ) -> str:
        ...


class GroqTranscriber:
    """Lazily configured Groq speech-to-text adapter."""

    def __init__(self) -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise TranscriptionConfigurationError(
                "GROQ_API_KEY is not configured."
            )

        try:
            from groq import Groq
        except ImportError as exc:
            raise TranscriptionConfigurationError(
                "Install the project's optional Groq dependency."
            ) from exc

        self._client = Groq(api_key=api_key)
        self._model = os.environ.get(
            "GROQ_TRANSCRIPTION_MODEL",
            "whisper-large-v3-turbo",
        )

    def transcribe(
        self,
        audio: bytes,
        *,
        filename: str,
        content_type: str,
    ) -> str:
        try:
            result = self._client.audio.transcriptions.create(
                file=(filename, audio, content_type),
                model=self._model,
                response_format="json",
                temperature=0.0,
                prompt=(
                    "Transcribe this field incident accurately. "
                    "Preserve spoken site IDs, asset IDs, equipment names, "
                    "and symptoms. Do not invent missing details."
                ),
            )
        except Exception as exc:
            raise TranscriptionError(
                "The transcription provider request failed."
            ) from exc

        transcript = getattr(result, "text", None)
        if not isinstance(transcript, str):
            raise TranscriptionError(
                "The transcription provider returned an invalid response."
            )

        return transcript.strip()
