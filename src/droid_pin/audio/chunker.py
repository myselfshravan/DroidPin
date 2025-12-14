"""Audio chunking for API compliance."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pydub import AudioSegment

from ..config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        logger.info(f"AudioChunker initialized: max_size={self.max_chunk_size_mb}MB, output={self.output_dir}")

    def get_file_size(self, file_path: Path) -> int:
        """Get file size in bytes."""
        return file_path.stat().st_size

    def needs_chunking(self, file_path: Path) -> bool:
        """Check if file exceeds size limit."""
        size = self.get_file_size(file_path)
        needs = size > self.max_chunk_size_bytes
        logger.info(f"File size: {size / (1024*1024):.2f}MB, needs chunking: {needs}")
        return needs

    def chunk_audio(
        self,
        audio_path: Path,
        progress_callback: Callable[[str], None] | None = None,
    ) -> list[AudioChunk]:
        """
        Split audio file into chunks under size limit.

        Uses a simple time-based approach: divide audio into ~5 minute chunks,
        which should comfortably fit under 15MB for most audio formats.

        Args:
            audio_path: Path to audio file.
            progress_callback: Optional callback for progress updates.

        Returns:
            List of AudioChunk objects.
        """
        def log(msg: str) -> None:
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        file_size = self.get_file_size(audio_path)
        log(f"Loading audio file: {audio_path.name} ({file_size / (1024*1024):.2f}MB)")

        # If file is small enough, return as single chunk without loading
        if not self.needs_chunking(audio_path):
            log("File is under size limit, using as single chunk")
            # Still need to get duration for metadata
            audio = AudioSegment.from_file(str(audio_path))
            duration_ms = len(audio)
            log(f"Audio duration: {duration_ms/1000:.1f}s")
            return [
                AudioChunk(
                    path=audio_path,
                    index=0,
                    start_ms=0,
                    end_ms=duration_ms,
                    duration_ms=duration_ms,
                )
            ]

        # Load audio for chunking
        log("File exceeds size limit, loading for chunking...")
        audio = AudioSegment.from_file(str(audio_path))
        total_duration_ms = len(audio)
        log(f"Audio duration: {total_duration_ms/1000:.1f}s ({total_duration_ms/60000:.1f} min)")

        # Use 5-minute chunks as a safe default (well under 15MB for most formats)
        chunk_duration_ms = 5 * 60 * 1000  # 5 minutes in ms

        # Calculate number of chunks needed
        num_chunks = (total_duration_ms + chunk_duration_ms - 1) // chunk_duration_ms
        log(f"Will create {num_chunks} chunks of ~{chunk_duration_ms/60000:.1f} min each")

        chunks: list[AudioChunk] = []

        for i in range(num_chunks):
            start_ms = i * chunk_duration_ms
            end_ms = min(start_ms + chunk_duration_ms, total_duration_ms)

            log(f"Creating chunk {i + 1}/{num_chunks}: {start_ms/1000:.1f}s - {end_ms/1000:.1f}s")

            chunk_audio = audio[start_ms:end_ms]
            chunk_path = self.output_dir / f"chunk_{i:04d}.mp3"

            # Export with compression
            chunk_audio.export(
                str(chunk_path),
                format="mp3",
                bitrate="64k",  # Lower bitrate for smaller files
                parameters=["-ac", "1"],  # Mono
            )

            chunk_size = self.get_file_size(chunk_path)
            log(f"Chunk {i + 1} exported: {chunk_size / (1024*1024):.2f}MB")

            # If chunk is still too large, split it further
            if chunk_size > self.max_chunk_size_bytes:
                log(f"Chunk {i + 1} too large, splitting further...")
                chunk_path.unlink()  # Remove oversized chunk

                # Split this segment in half recursively
                sub_chunks = self._split_segment(
                    audio, start_ms, end_ms, len(chunks), log
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(
                    AudioChunk(
                        path=chunk_path,
                        index=len(chunks),
                        start_ms=start_ms,
                        end_ms=end_ms,
                        duration_ms=end_ms - start_ms,
                    )
                )

        log(f"Chunking complete: {len(chunks)} chunks created")
        return chunks

    def _split_segment(
        self,
        audio: AudioSegment,
        start_ms: int,
        end_ms: int,
        base_index: int,
        log: Callable[[str], None],
    ) -> list[AudioChunk]:
        """Recursively split a segment until chunks are under size limit."""
        mid_ms = (start_ms + end_ms) // 2

        # Prevent infinite recursion with minimum chunk size (10 seconds)
        if mid_ms - start_ms < 10_000:
            log(f"Warning: Chunk at {start_ms/1000:.1f}s is at minimum size")
            chunk_audio = audio[start_ms:end_ms]
            chunk_path = self.output_dir / f"chunk_{base_index:04d}.mp3"
            chunk_audio.export(str(chunk_path), format="mp3", bitrate="32k", parameters=["-ac", "1"])
            return [
                AudioChunk(
                    path=chunk_path,
                    index=base_index,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    duration_ms=end_ms - start_ms,
                )
            ]

        chunks = []
        for seg_start, seg_end in [(start_ms, mid_ms), (mid_ms, end_ms)]:
            chunk_audio = audio[seg_start:seg_end]
            chunk_path = self.output_dir / f"chunk_{base_index + len(chunks):04d}.mp3"
            chunk_audio.export(str(chunk_path), format="mp3", bitrate="64k", parameters=["-ac", "1"])

            chunk_size = self.get_file_size(chunk_path)

            if chunk_size > self.max_chunk_size_bytes:
                chunk_path.unlink()
                sub_chunks = self._split_segment(
                    audio, seg_start, seg_end, base_index + len(chunks), log
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(
                    AudioChunk(
                        path=chunk_path,
                        index=base_index + len(chunks),
                        start_ms=seg_start,
                        end_ms=seg_end,
                        duration_ms=seg_end - seg_start,
                    )
                )

        return chunks

    def cleanup_chunks(self, chunks: list[AudioChunk]) -> None:
        """Remove temporary chunk files."""
        for chunk in chunks:
            if chunk.path.exists() and "chunk_" in chunk.path.name:
                chunk.path.unlink()
                logger.info(f"Cleaned up {chunk.path.name}")
