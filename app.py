"""Streamlit entry point for the AI Resume Screening Agent."""

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="AI Resume Screening Agent", page_icon="🧠", layout="wide")

st.title("AI Resume Screening Agent")
st.caption("Explainable NLP-based candidate ranking and decision support")

st.info(
    "This system is a decision-support tool. It should not automatically reject candidates or replace human hiring judgment."
)

st.write("Phase 1 scaffold complete. The full screening workflow will be implemented in subsequent phases.")
