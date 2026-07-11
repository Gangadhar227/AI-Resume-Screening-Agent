from __future__ import annotations

import pytest

from src.similarity_engine import ModelLoadError, SemanticSimilarityEngine, compute_similarity


class DummyEmbeddingModel:
    def __init__(self, *args, **kwargs) -> None:
        self.calls = 0

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        self.calls += 1
        if isinstance(texts, list):
            return [[1.0, 0.0] if text == "valid" else [0.0, 1.0] for text in texts]
        return [1.0, 0.0]


def test_empty_job_description_raises_value_error() -> None:
    with pytest.raises(ValueError, match="job description"):
        compute_similarity("   ", ["resume text"])


def test_empty_resume_validation() -> None:
    with pytest.raises(ValueError, match="resume text"):
        compute_similarity("job description", ["   "])


def test_empty_batch_validation() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_similarity("job description", [])


def test_valid_semantic_score_uses_mocked_model(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
            return [[1.0, 0.0] for _ in texts]

    def fake_loader(*args, **kwargs):
        return FakeModel()

    monkeypatch.setattr("src.similarity_engine._load_model", fake_loader)
    engine = SemanticSimilarityEngine(model_name="dummy")
    scores = engine.compute_similarity_batch("valid", ["valid", "other"])
    assert scores[0] == 100.0
    assert scores[1] == 0.0


def test_score_range_is_clamped_to_0_100(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("src.similarity_engine._load_model", lambda *args, **kwargs: FakeModel())
    engine = SemanticSimilarityEngine(model_name="dummy")
    score = engine.compute_similarity("valid", "valid")
    assert 0 <= score <= 100


def test_model_loading_failure_raises_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_loader(*args, **kwargs):
        raise RuntimeError("download failed")

    monkeypatch.setattr("src.similarity_engine._load_model", fake_loader)

    with pytest.raises(ModelLoadError, match="download"):
        SemanticSimilarityEngine(model_name="dummy")
