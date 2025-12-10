"""Groq Whisper API client."""

import time
from dataclasses import dataclass, field
from pathlib import Path

from groq import Groq
from groq._exceptions import APIError, RateLimitError

from ..config import settings


@dataclass
class TranscriptionResult:
    """Result from a single transcription request."""

    text: str
    language: str | None = None
    duration: float | None = None
    segments: list | None = field(default_factory=list)
    words: list | None = field(default_factory=list)


class GroqWhisperClient:
    """Wrapper for Groq Whisper API with retry logic."""

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialize the Groq Whisper client.

        Args:
            api_key: Groq API key. Defaults to environment variable.

        Raises:
            ValueError: If no API key is provided.
        """
        self.api_key = api_key or settings.groq_api_key
        if not self.api_key:
            raise ValueError("Groq API key not provided")
        self.client = Groq(api_key=self.api_key)
        self.config = settings.transcription

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
        prompt: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> TranscriptionResult:
        """
        Transcribe audio file using Groq Whisper API.

        Args:
            audio_path: Path to audio file.
            language: ISO-639-1 language code (optional, auto-detect if None).
            prompt: Context prompt for transcription style.
            max_retries: Maximum retry attempts on failure.
            retry_delay: Delay between retries in seconds.

        Returns:
            TranscriptionResult with text and metadata.

        Raises:
            APIError: If transcription fails after retries.
        """
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                with open(audio_path, "rb") as audio_file:
                    response = self.client.audio.transcriptions.create(
                        file=(audio_path.name, audio_file),
                        model=self.config.model,
                        language=language or self.config.language,
                        response_format=self.config.response_format,
                        temperature=self.config.temperature,
                        prompt=prompt,
                    )

                # Parse response based on format
                if self.config.response_format == "text":
                    return TranscriptionResult(text=str(response))
                elif self.config.response_format == "verbose_json":
                    return TranscriptionResult(
                        text=response.text,
                        language=getattr(response, "language", None),
                        duration=getattr(response, "duration", None),
                        segments=getattr(response, "segments", None),
                        words=getattr(response, "words", None),
                    )
                else:  # json
                    return TranscriptionResult(text=response.text)

            except RateLimitError as e:
                last_error = e
                wait_time = retry_delay * (2**attempt)  # Exponential backoff
                time.sleep(wait_time)

            except APIError as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        raise last_error or Exception("Transcription failed after retries")
