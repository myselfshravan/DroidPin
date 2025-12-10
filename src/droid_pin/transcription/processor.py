"""Transcription orchestration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..audio.chunker import AudioChunk, AudioChunker
from ..audio.extractor import AudioExtractor
from .client import GroqWhisperClient, TranscriptionResult


@dataclass
class ProcessingResult:
    """Complete processing result."""

    text: str
    language: str | None = None
    total_duration_ms: int = 0
    chunk_count: int = 1
    segments: list = field(default_factory=list)
    source_file: str = ""
    was_video: bool = False


class TranscriptionProcessor:
    """Orchestrates the complete transcription workflow."""

    def __init__(
        self,
        api_key: str | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> None:
        """
        Initialize the transcription processor.

        Args:
            api_key: Groq API key.
            progress_callback: Callback(current, total, message) for progress updates.
        """
        self.extractor = AudioExtractor()
        self.chunker = AudioChunker()
        self.client = GroqWhisperClient(api_key)
        self.progress_callback = progress_callback

    def _report_progress(self, current: int, total: int, message: str) -> None:
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(current, total, message)

    def _combine_transcriptions(
        self,
        results: list[TranscriptionResult],
        chunks: list[AudioChunk],
    ) -> str:
        """
        Combine chunk transcriptions.

        Simple approach: concatenate with space separation.
        Could be enhanced with overlap detection/deduplication.

        Args:
            results: List of transcription results.
            chunks: List of audio chunks.

        Returns:
            Combined transcription text.
        """
        texts = [r.text.strip() for r in results if r.text]
        return " ".join(texts)

    def process(
        self,
        file_path: Path,
        language: str | None = None,
        prompt: str | None = None,
    ) -> ProcessingResult:
        """
        Process audio or video file for transcription.

        Args:
            file_path: Path to audio or video file.
            language: ISO-639-1 language code (optional).
            prompt: Context prompt for transcription.

        Returns:
            ProcessingResult with combined transcription.
        """
        audio_path = file_path
        was_video = False
        temp_audio_path: Path | None = None
        chunks: list[AudioChunk] = []

        try:
            # Step 1: Extract audio if video
            if self.extractor.is_video_file(file_path):
                self._report_progress(0, 100, "Extracting audio from video...")
                audio_path = self.extractor.extract(file_path)
                temp_audio_path = audio_path
                was_video = True
                self._report_progress(10, 100, "Audio extracted")

            # Step 2: Chunk audio if needed
            self._report_progress(15, 100, "Analyzing audio file...")
            chunks = self.chunker.chunk_audio(audio_path)
            chunk_count = len(chunks)
            self._report_progress(20, 100, f"Split into {chunk_count} chunk(s)")

            # Step 3: Transcribe each chunk
            results: list[TranscriptionResult] = []
            for i, chunk in enumerate(chunks):
                progress = 20 + int((i / chunk_count) * 70)
                self._report_progress(
                    progress,
                    100,
                    f"Transcribing chunk {i + 1}/{chunk_count}...",
                )

                result = self.client.transcribe(
                    audio_path=chunk.path,
                    language=language,
                    prompt=prompt,
                )
                results.append(result)

            # Step 4: Combine results
            self._report_progress(90, 100, "Combining transcriptions...")
            combined_text = self._combine_transcriptions(results, chunks)

            # Aggregate metadata
            total_duration = sum(c.duration_ms for c in chunks)
            detected_language = results[0].language if results else None
            all_segments = []
            for result in results:
                if result.segments:
                    all_segments.extend(result.segments)

            self._report_progress(100, 100, "Transcription complete!")

            return ProcessingResult(
                text=combined_text,
                language=detected_language,
                total_duration_ms=total_duration,
                chunk_count=chunk_count,
                segments=all_segments,
                source_file=file_path.name,
                was_video=was_video,
            )

        finally:
            # Cleanup temporary files
            self.chunker.cleanup_chunks(chunks)
            if temp_audio_path:
                self.extractor.cleanup(temp_audio_path)
