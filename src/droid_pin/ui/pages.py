"""Page layouts for the Streamlit application."""

import streamlit as st

from ..config import settings
from ..transcription.processor import TranscriptionProcessor
from . import components


def main_page() -> None:
    """Main transcription page."""
    st.title("Audio/Video Transcription")
    st.markdown("Powered by Groq Whisper API (whisper-large-v3)")

    # Validate configuration
    errors = settings.validate()
    if errors:
        for error in errors:
            st.error(error)
        st.info("Please set the GROQ_API_KEY environment variable and restart.")
        st.code("export GROQ_API_KEY=your_api_key_here", language="bash")
        return

    # Sidebar options
    options = components.transcription_options()

    # File upload
    file_path = components.file_uploader()

    if file_path is None:
        st.info("Upload an audio or video file to get started")
        return

    # Show file info
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    st.info(f"File: {file_path.name} ({file_size_mb:.2f} MB)")

    # Transcribe button
    if st.button("Transcribe", type="primary", use_container_width=True):
        progress_update, progress_bar, status_text = components.progress_indicator()

        try:
            processor = TranscriptionProcessor(progress_callback=progress_update)

            result = processor.process(
                file_path=file_path,
                language=options["language"],
                prompt=options["prompt"],
            )

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            # Display result
            components.display_result(result)

        except Exception as e:
            st.error(f"Transcription failed: {e!s}")

        finally:
            # Cleanup uploaded file
            if file_path.exists():
                file_path.unlink()
