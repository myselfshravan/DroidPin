"""Configuration management for the transcription application."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass(frozen=True)
class AudioConfig:
    """Audio processing configuration."""

    max_chunk_size_mb: float = 15.0  # Groq free tier is 25MB, use 15MB for safety
    sample_rate: int = 16000  # Optimal for Whisper
    channels: int = 1  # Mono
    export_format: str = "mp3"  # Good compression, widely supported

    @property
    def max_chunk_size_bytes(self) -> int:
        """Maximum chunk size in bytes."""
        return int(self.max_chunk_size_mb * 1024 * 1024)


@dataclass(frozen=True)
class TranscriptionConfig:
    """Groq API transcription configuration."""

    model: Literal["whisper-large-v3", "whisper-large-v3-turbo"] = "whisper-large-v3"
    language: str | None = None  # Auto-detect if None
    response_format: Literal["json", "text", "verbose_json"] = "verbose_json"
    temperature: float = 0.0


@dataclass(frozen=True)
class FileConfig:
    """Supported file formats."""

    audio_extensions: tuple[str, ...] = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm")
    video_extensions: tuple[str, ...] = (".mp4", ".webm", ".mkv", ".avi", ".mov", ".wmv")

    @property
    def all_extensions(self) -> tuple[str, ...]:
        """All supported file extensions."""
        return self.audio_extensions + self.video_extensions


class Settings:
    """Application settings with environment variable support."""

    def __init__(self) -> None:
        self.groq_api_key: str = os.getenv("GROQ_API_KEY", "")
        self.audio = AudioConfig()
        self.transcription = TranscriptionConfig()
        self.files = FileConfig()
        self.temp_dir: Path = Path(os.getenv("TEMP_DIR", "/tmp/droid-pin"))

    def validate(self) -> list[str]:
        """Validate configuration, return list of errors."""
        errors = []
        if not self.groq_api_key:
            errors.append("GROQ_API_KEY environment variable not set")
        return errors


# Singleton instance
settings = Settings()
