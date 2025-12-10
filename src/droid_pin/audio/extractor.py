"""Extract audio from video files."""

from pathlib import Path

from moviepy import VideoFileClip

from ..config import settings


class AudioExtractor:
    """Extract audio from video files using MoviePy."""

    SUPPORTED_VIDEO_FORMATS = settings.files.video_extensions

    def __init__(self, output_dir: Path | None = None) -> None:
        """
        Initialize the audio extractor.

        Args:
            output_dir: Directory for extracted audio files. Defaults to temp_dir.
        """
        self.output_dir = output_dir or settings.temp_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def is_video_file(self, file_path: Path) -> bool:
        """
        Check if file is a supported video format.

        Args:
            file_path: Path to the file to check.

        Returns:
            True if file is a supported video format.
        """
        return file_path.suffix.lower() in self.SUPPORTED_VIDEO_FORMATS

    def extract(
        self,
        video_path: Path,
        output_format: str = "mp3",
        sample_rate: int | None = None,
    ) -> Path:
        """
        Extract audio from video file.

        Args:
            video_path: Path to video file.
            output_format: Audio format (mp3, wav, flac).
            sample_rate: Sample rate for output audio. Defaults to config value.

        Returns:
            Path to extracted audio file.

        Raises:
            ValueError: If video has no audio track.
        """
        sample_rate = sample_rate or settings.audio.sample_rate
        output_path = self.output_dir / f"{video_path.stem}.{output_format}"

        video_clip = None
        try:
            video_clip = VideoFileClip(str(video_path))

            if video_clip.audio is None:
                raise ValueError(f"Video file has no audio track: {video_path}")

            video_clip.audio.write_audiofile(
                str(output_path),
                fps=sample_rate,
                nbytes=2,  # 16-bit audio
                codec="libmp3lame" if output_format == "mp3" else None,
                verbose=False,
                logger=None,
            )

            return output_path

        finally:
            if video_clip:
                video_clip.close()

    def cleanup(self, file_path: Path) -> None:
        """
        Remove temporary file.

        Args:
            file_path: Path to the file to remove.
        """
        if file_path.exists():
            file_path.unlink()
