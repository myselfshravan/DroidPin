"""SQLite database for storing transcription results."""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionRecord:
    """A stored transcription record."""

    id: int
    filename: str
    text: str
    language: Optional[str]
    duration_ms: int
    chunk_count: int
    created_at: datetime
    file_hash: Optional[str] = None
    segments: Optional[list] = None

    @property
    def duration_formatted(self) -> str:
        """Return duration as human-readable string."""
        seconds = self.duration_ms / 1000
        if seconds >= 3600:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
        elif seconds >= 60:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            return f"{seconds:.1f}s"


class TranscriptionDB:
    """SQLite database for transcription storage."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """
        Initialize the database.

        Args:
            db_path: Path to SQLite database file. Defaults to ~/.droid-pin/transcriptions.db
        """
        if db_path is None:
            db_dir = Path.home() / ".droid-pin"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "transcriptions.db"

        self.db_path = db_path
        self._init_db()
        logger.info(f"TranscriptionDB initialized at {self.db_path}")

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transcriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    text TEXT NOT NULL,
                    language TEXT,
                    duration_ms INTEGER,
                    chunk_count INTEGER,
                    file_hash TEXT,
                    segments TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_filename ON transcriptions(filename)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON transcriptions(created_at)
            """)
            conn.commit()

    def save(
        self,
        filename: str,
        text: str,
        language: Optional[str] = None,
        duration_ms: int = 0,
        chunk_count: int = 1,
        file_hash: Optional[str] = None,
        segments: Optional[list] = None,
    ) -> int:
        """
        Save a transcription to the database.

        Args:
            filename: Original filename.
            text: Transcription text.
            language: Detected language code.
            duration_ms: Audio duration in milliseconds.
            chunk_count: Number of chunks processed.
            file_hash: Optional hash of the source file.
            segments: Optional list of transcription segments.

        Returns:
            ID of the inserted record.
        """
        segments_json = json.dumps(segments) if segments else None

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO transcriptions
                (filename, text, language, duration_ms, chunk_count, file_hash, segments)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (filename, text, language, duration_ms, chunk_count, file_hash, segments_json),
            )
            conn.commit()
            record_id = cursor.lastrowid
            logger.info(f"Saved transcription #{record_id}: {filename}")
            return record_id

    def get(self, record_id: int) -> Optional[TranscriptionRecord]:
        """
        Get a transcription by ID.

        Args:
            record_id: The record ID.

        Returns:
            TranscriptionRecord or None if not found.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM transcriptions WHERE id = ?",
                (record_id,),
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return self._row_to_record(row)

    def get_by_filename(self, filename: str) -> list[TranscriptionRecord]:
        """
        Get all transcriptions for a filename.

        Args:
            filename: The filename to search for.

        Returns:
            List of matching TranscriptionRecords.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM transcriptions WHERE filename = ? ORDER BY created_at DESC",
                (filename,),
            )
            return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_recent(self, limit: int = 20) -> list[TranscriptionRecord]:
        """
        Get recent transcriptions.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of TranscriptionRecords, most recent first.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM transcriptions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            return [self._row_to_record(row) for row in cursor.fetchall()]

    def search(self, query: str, limit: int = 50) -> list[TranscriptionRecord]:
        """
        Search transcriptions by text content.

        Args:
            query: Search query string.
            limit: Maximum number of results.

        Returns:
            List of matching TranscriptionRecords.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM transcriptions
                WHERE text LIKE ? OR filename LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", limit),
            )
            return [self._row_to_record(row) for row in cursor.fetchall()]

    def delete(self, record_id: int) -> bool:
        """
        Delete a transcription by ID.

        Args:
            record_id: The record ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM transcriptions WHERE id = ?",
                (record_id,),
            )
            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted transcription #{record_id}")
            return deleted

    def get_stats(self) -> dict:
        """
        Get database statistics.

        Returns:
            Dict with count, total_duration, etc.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as count,
                    COALESCE(SUM(duration_ms), 0) as total_duration_ms,
                    COALESCE(SUM(LENGTH(text)), 0) as total_chars
                FROM transcriptions
            """)
            row = cursor.fetchone()
            return {
                "count": row[0],
                "total_duration_ms": row[1],
                "total_chars": row[2],
            }

    def _row_to_record(self, row: sqlite3.Row) -> TranscriptionRecord:
        """Convert a database row to a TranscriptionRecord."""
        segments = None
        if row["segments"]:
            try:
                segments = json.loads(row["segments"])
            except json.JSONDecodeError:
                pass

        return TranscriptionRecord(
            id=row["id"],
            filename=row["filename"],
            text=row["text"],
            language=row["language"],
            duration_ms=row["duration_ms"] or 0,
            chunk_count=row["chunk_count"] or 1,
            created_at=datetime.fromisoformat(row["created_at"]),
            file_hash=row["file_hash"],
            segments=segments,
        )
