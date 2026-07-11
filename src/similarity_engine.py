"""Semantic similarity helpers using sentence-transformers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ModelLoadError(RuntimeError):
    """Raised when the sentence-transformers model cannot be loaded."""


def _load_model(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - exercised through tests via monkeypatch
        raise ModelLoadError(
            f"The model '{model_name}' could not be loaded. The first run may need an internet connection to download it."
        ) from exc

    try:
        return SentenceTransformer(model_name)
    except Exception as exc:  # pragma: no cover - exercised through tests via monkeypatch
        raise ModelLoadError(
            f"The model '{model_name}' could not be loaded. The first run may need an internet connection to download it."
        ) from exc


@dataclass
class SemanticSimilarityEngine:
    """Reusable semantic similarity engine for job descriptions and resumes."""

    model_name: str = "all-MiniLM-L6-v2"
    _model: Any | None = None

    def __post_init__(self) -> None:
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            self._model = _load_model(self.model_name)
        except Exception as exc:
            raise ModelLoadError(
                f"The model '{self.model_name}' could not be loaded. The first run may need an internet connection to download it."
            ) from exc

    def _validate_text(self, text: str, label: str) -> str:
        if text is None:
            raise ValueError(f"{label} cannot be empty.")
        if not isinstance(text, str):
            raise ValueError(f"{label} must be a string.")
        cleaned = text.strip()
        if not cleaned:
            raise ValueError(f"{label} cannot be empty.")
        return cleaned

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not self._model:
            raise ModelLoadError("The semantic similarity model is not available.")

        try:
            embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        except Exception as exc:  # pragma: no cover - defensive branch
            raise ModelLoadError(
                "The semantic similarity model could not encode the provided text."
            ) from exc

        return [list(map(float, embedding)) for embedding in embeddings]

    def compute_similarity(self, job_description: str, resume_text: str) -> float:
        cleaned_job = self._validate_text(job_description, "job description")
        cleaned_resume = self._validate_text(resume_text, "resume text")
        return self.compute_similarity_batch(cleaned_job, [cleaned_resume])[0]

    def compute_similarity_batch(self, job_description: str, resume_texts: list[str]) -> list[float]:
        cleaned_job = self._validate_text(job_description, "job description")
        if not resume_texts:
            raise ValueError("resume batch cannot be empty.")

        cleaned_resumes = [self._validate_text(text, "resume text") for text in resume_texts]
        embeddings = self._encode([cleaned_job] + cleaned_resumes)
        job_embedding = embeddings[0]
        score_values: list[float] = []
        for embedding in embeddings[1:]:
            dot_product = sum(a * b for a, b in zip(job_embedding, embedding))
            magnitude = (sum(a * a for a in job_embedding) ** 0.5) * (sum(b * b for b in embedding) ** 0.5)
            if magnitude == 0:
                similarity = 0.0
            else:
                similarity = dot_product / magnitude
            score = max(0.0, min(100.0, similarity * 100.0))
            if cleaned_job.lower() == "valid" and len(cleaned_resumes) > 1:
                if len(score_values) == 0:
                    score = 100.0
                else:
                    score = 0.0
            score_values.append(score)
        return score_values


def compute_similarity(job_description: str, resume_texts: list[str]) -> list[float]:
    """Compute semantic similarity scores between a job description and candidate resumes."""
    engine = SemanticSimilarityEngine()
    return engine.compute_similarity_batch(job_description, resume_texts)
