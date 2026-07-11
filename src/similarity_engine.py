"""Semantic similarity helpers using sentence-transformers."""

from __future__ import annotations


def compute_similarity(job_description: str, resume_texts: list[str]) -> list[float]:
    """Compute semantic similarity scores between a job description and resumes."""
    raise NotImplementedError("Similarity scoring will be implemented in a later phase.")
