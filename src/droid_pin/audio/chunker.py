"""Audio chunking for API compliance."""

from dataclasses import dataclass
from pathlib import Path

from pydub import AudioSegment

from ..config import settings


@dataclass
class AudioChunk:
    """Represents a chunk of audio."""

    path: Path
    index: int
    start_ms: int
    end_ms: int
    duration_ms: int


class AudioChunker:
    """Split audio files into API-compliant chunks."""

    def __init__(
        self,
        max_chunk_size_mb: float | None = None,
        output_dir: Path | None = None,
    ) -> None:
        """
        Initialize the audio chunker.

        Args:
            max_chunk_size_mb: Maximum chunk size in MB. Defaults to config value.
            output_dir: Directory for chunk files. Defaults to temp_dir.
        """
        self.max_chunk_size_mb = max_chunk_size_mb or settings.audio.max_chunk_size_mb
        self.max_chunk_size_bytes = int(self.max_chunk_size_mb * 1024 * 1024)
        self.output_dir = output_dir or settings.temp_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_file_size(self, file_path: Path) -> int:
        """
        Get file size in bytes.

        Args:
            file_path: Path to the file.

        Returns:
            File size in bytes.
        """
        return file_path.stat().st_size

    def needs_chunking(self, file_path: Path) -> bool:
        """
        Check if file exceeds size limit.

        Args:
            file_path: Path to the file.

        Returns:
            True if file needs to be chunked.
        """
        return self.get_file_size(file_path) > self.max_chunk_size_bytes

    def _estimate_chunk_duration(self, audio: AudioSegment, file_size: int) -> int:
        """
        Estimate chunk duration in ms to stay under size limit.

        Args:
            audio: The audio segment.
            file_size: Size of the original file in bytes.

        Returns:
            Estimated chunk duration in milliseconds.
        """
        duration_ms = len(audio)
        if duration_ms == 0:
            return 60_000  # Default to 1 minute

        # Calculate bytes per ms from actual file
        bytes_per_ms = file_size / duration_ms

        # Target 80% of max size for safety margin
        target_bytes = self.max_chunk_size_bytes * 0.8
        chunk_duration_ms = int(target_bytes / bytes_per_ms)

        # Minimum 30 seconds, maximum 10 minutes
        return max(30_000, min(chunk_duration_ms, 600_000))

    def chunk_audio(
        self,
        audio_path: Path,
        overlap_ms: int = 1000,
    ) -> list[AudioChunk]:
        """
        Split audio file into chunks under size limit.

        Args:
            audio_path: Path to audio file.
            overlap_ms: Overlap between chunks to prevent word splitting.

        Returns:
            List of AudioChunk objects.
        """
        file_size = self.get_file_size(audio_path)

        # If file is small enough, return as single chunk
        if not self.needs_chunking(audio_path):
            audio = AudioSegment.from_file(str(audio_path))
            return [
                AudioChunk(
                    path=audio_path,
                    index=0,
                    start_ms=0,
                    end_ms=len(audio),
                    duration_ms=len(audio),
                )
            ]

        audio = AudioSegment.from_file(str(audio_path))
        chunk_duration_ms = self._estimate_chunk_duration(audio, file_size)
        chunks: list[AudioChunk] = []

        current_start = 0
        chunk_index = 0

        while current_start < len(audio):
            current_end = min(current_start + chunk_duration_ms, len(audio))
            chunk_audio = audio[current_start:current_end]

            # Export chunk to temporary file
            chunk_path = self.output_dir / f"chunk_{chunk_index:04d}.mp3"
            chunk_audio.export(
                str(chunk_path),
                format="mp3",
                parameters=["-ar", str(settings.audio.sample_rate), "-ac", "1"],
            )

            # Verify chunk size, reduce duration if needed
            while self.get_file_size(chunk_path) > self.max_chunk_size_bytes:
                chunk_duration_ms = int(chunk_duration_ms * 0.8)
                current_end = min(current_start + chunk_duration_ms, len(audio))
                chunk_audio = audio[current_start:current_end]
                chunk_audio.export(
                    str(chunk_path),
                    format="mp3",
                    parameters=["-ar", str(settings.audio.sample_rate), "-ac", "1"],
                )

            chunks.append(
                AudioChunk(
                    path=chunk_path,
                    index=chunk_index,
                    start_ms=current_start,
                    end_ms=current_end,
                    duration_ms=current_end - current_start,
                )
            )

            # Move to next chunk with overlap
            current_start = current_end - overlap_ms
            chunk_index += 1

        return chunks

    def cleanup_chunks(self, chunks: list[AudioChunk]) -> None:
        """
        Remove temporary chunk files.

        Args:
            chunks: List of chunks to clean up.
        """
        for chunk in chunks:
            if chunk.path.exists() and "chunk_" in chunk.path.name:
                chunk.path.unlink()
