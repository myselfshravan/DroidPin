"""Streamlit application entry point."""

import streamlit as st
from src.droid_pin.ui.pages import main_page

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Transcription App",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="auto",
)


def main() -> None:
    """Run the Streamlit application."""
    main_page()


if __name__ == "__main__":
    main()
