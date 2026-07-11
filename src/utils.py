"""Utility helpers for the AI Resume Screening Agent."""

from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """Return the repository root path."""
    return Path(__file__).resolve().parents[1]
