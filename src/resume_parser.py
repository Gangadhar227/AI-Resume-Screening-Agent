"""Resume parsing helpers for PDF, DOCX, and TXT files."""

from __future__ import annotations

from pathlib import Path


def extract_text_from_file(file_path: str | Path) -> str:
    """Extract text from a supported resume file."""
    raise NotImplementedError("Resume parsing will be implemented in a later phase.")
