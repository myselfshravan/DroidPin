"""Streamlit application entry point."""

import streamlit as st

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Transcription App",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="auto",
)

from src.droid_pin.ui.pages import main_page


def main() -> None:
    """Run the Streamlit application."""
    main_page()


if __name__ == "__main__":
    main()
