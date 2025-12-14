"""Page layouts for the Streamlit application."""

import streamlit as st

from ..config import settings
from ..storage.database import TranscriptionDB
from ..transcription.processor import TranscriptionProcessor
from . import components


def main_page() -> None:
    """Main transcription page with tabs for transcribe and history."""
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

    # Initialize database
    db = TranscriptionDB()

    # Create tabs
    tab_transcribe, tab_history = st.tabs(["Transcribe", "History"])

    with tab_transcribe:
        _transcribe_tab(db)

    with tab_history:
        _history_tab(db)


def _transcribe_tab(db: TranscriptionDB) -> None:
    """Transcription tab content."""
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

            # Save to database
            record_id = db.save(
                filename=result.source_file,
                text=result.text,
                language=result.language,
                duration_ms=result.total_duration_ms,
                chunk_count=result.chunk_count,
                segments=result.segments,
            )
            st.success(f"Transcription saved to local database (ID: {record_id})")

            # Display result
            components.display_result(result)

        except Exception as e:
            st.error(f"Transcription failed: {e!s}")

        finally:
            # Cleanup uploaded file
            if file_path.exists():
                file_path.unlink()


def _history_tab(db: TranscriptionDB) -> None:
    """History tab content."""
    # Stats
    stats = db.get_stats()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transcriptions", stats["count"])
    with col2:
        hours = stats["total_duration_ms"] / (1000 * 60 * 60)
        st.metric("Total Audio", f"{hours:.1f} hours")
    with col3:
        chars = stats["total_chars"]
        st.metric("Total Characters", f"{chars:,}")

    st.divider()

    # Search
    search_query = st.text_input("Search transcriptions", placeholder="Enter keywords...")

    if search_query:
        records = db.search(search_query)
        st.caption(f"Found {len(records)} results")
    else:
        records = db.get_recent(limit=20)
        st.caption("Recent transcriptions")

    if not records:
        st.info("No transcriptions yet. Upload a file to get started!")
        return

    # Display records
    for record in records:
        with st.expander(
            f"**{record.filename}** - {record.duration_formatted} - {record.created_at.strftime('%Y-%m-%d %H:%M')}"
        ):
            # Metadata row
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            with meta_col1:
                st.caption(f"Language: {(record.language or 'N/A').upper()}")
            with meta_col2:
                st.caption(f"Chunks: {record.chunk_count}")
            with meta_col3:
                st.caption(f"ID: {record.id}")

            # Text preview
            st.text_area(
                "Transcription",
                value=record.text,
                height=200,
                key=f"text_{record.id}",
                label_visibility="collapsed",
            )

            # Action buttons
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                st.download_button(
                    "Download",
                    data=record.text,
                    file_name=f"{record.filename}_transcription.txt",
                    mime="text/plain",
                    key=f"download_{record.id}",
                )
            with btn_col2:
                if st.button("Copy to Clipboard", key=f"copy_{record.id}"):
                    st.code(record.text[:500] + "..." if len(record.text) > 500 else record.text)
            with btn_col3:
                if st.button("Delete", key=f"delete_{record.id}", type="secondary"):
                    db.delete(record.id)
                    st.rerun()
