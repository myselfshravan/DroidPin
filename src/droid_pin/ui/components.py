"""Reusable Streamlit UI components."""

from pathlib import Path
from typing import Callable

import streamlit as st

from ..config import settings
from ..transcription.processor import ProcessingResult


def file_uploader() -> Path | None:
    """
    File uploader component for audio/video files.

    Returns:
        Path to temporary file, or None if no file uploaded.
    """
    all_extensions = [ext.lstrip(".") for ext in settings.files.all_extensions]

    uploaded_file = st.file_uploader(
        "Upload an audio or video file",
        type=all_extensions,
        help=f"Supported formats: {', '.join(all_extensions)}",
    )

    if uploaded_file is None:
        return None

    # Save to temporary file (required for pydub/moviepy)
    temp_dir = settings.temp_dir
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_path = temp_dir / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return temp_path


def transcription_options() -> dict:
    """
    Sidebar options for transcription configuration.

    Returns:
        Dict with language and prompt options.
    """
    with st.sidebar:
        st.header("Options")

        language = st.selectbox(
            "Language",
            options=[None, "en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko", "hi"],
            format_func=lambda x: "Auto-detect" if x is None else x.upper(),
            help="Select language or auto-detect",
        )

        prompt = st.text_area(
            "Context prompt (optional)",
            placeholder="E.g., technical terms, speaker names...",
            help="Provide context for better transcription accuracy",
        )

        st.divider()

        st.caption("Model: whisper-large-v3")
        st.caption(f"Max chunk size: {settings.audio.max_chunk_size_mb}MB")

    return {
        "language": language,
        "prompt": prompt if prompt else None,
    }


def progress_indicator() -> tuple[Callable[[int, int, str], None], st.delta_generator.DeltaGenerator, st.delta_generator.DeltaGenerator]:
    """
    Create a progress indicator that can be updated.

    Returns:
        Tuple of (update_function, progress_bar, status_text).
    """
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update(current: int, total: int, message: str) -> None:
        progress_bar.progress(current / total)
        status_text.text(message)

    return update, progress_bar, status_text


def display_result(result: ProcessingResult) -> None:
    """
    Display transcription result with formatting.

    Args:
        result: The processing result to display.
    """
    st.success("Transcription Complete!")

    # Metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        duration_sec = result.total_duration_ms / 1000
        if duration_sec >= 60:
            st.metric("Duration", f"{duration_sec / 60:.1f} min")
        else:
            st.metric("Duration", f"{duration_sec:.1f} sec")
    with col2:
        st.metric("Chunks", result.chunk_count)
    with col3:
        st.metric("Language", (result.language or "N/A").upper())

    # Transcription text
    st.subheader("Transcription")
    st.text_area(
        "Result",
        value=result.text,
        height=300,
        label_visibility="collapsed",
    )

    # Download button
    st.download_button(
        label="Download Transcription",
        data=result.text,
        file_name=f"{Path(result.source_file).stem}_transcription.txt",
        mime="text/plain",
    )
