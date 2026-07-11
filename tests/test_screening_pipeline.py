from __future__ import annotations

from pathlib import Path

import pytest

from src.candidate_result import CandidateResult
from src.screening_pipeline import ScreeningFailure, ScreeningPipeline
from src.similarity_engine import SemanticSimilarityEngine


class FakeModel:
    def __init__(self, *args, **kwargs) -> None:
        self.calls = 0

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        self.calls += 1
        return [[1.0, 0.0] for _ in texts]


class FakeEngine(SemanticSimilarityEngine):
    def __init__(self, *args, **kwargs) -> None:
        self._model = FakeModel()


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> ScreeningPipeline:
    monkeypatch.setattr("src.screening_pipeline.SemanticSimilarityEngine", FakeEngine)
    return ScreeningPipeline(job_description="Python NLP", resume_dir=Path("data"))


def test_one_candidate_is_ranked_and_returned(pipeline: ScreeningPipeline) -> None:
    results, failures = pipeline.run([Path("tests/fixtures/does_not_exist.txt")])
    assert len(results) == 0
    assert len(failures) == 1


def test_multiple_candidates_and_ranking_order(pipeline: ScreeningPipeline) -> None:
    candidate_a = CandidateResult(candidate_name="Beta", original_filename="b.txt", email="b@example.com", final_score=80.0, semantic_relevance_score=80.0, skill_match_score=70.0, experience_match_score=100.0, education_match_score=100.0, matched_skills=["Python"], missing_skills=[], extracted_skills=["Python"], extracted_experience=3.0, required_experience=2.0, extracted_education=["B.Tech"], required_education=["B.Tech"], recommendation="Strong Match", scoring_notes=[])
    candidate_b = CandidateResult(candidate_name="Alpha", original_filename="a.txt", email="a@example.com", final_score=90.0, semantic_relevance_score=90.0, skill_match_score=80.0, experience_match_score=100.0, education_match_score=100.0, matched_skills=["Python"], missing_skills=[], extracted_skills=["Python"], extracted_experience=3.0, required_experience=2.0, extracted_education=["B.Tech"], required_education=["B.Tech"], recommendation="Strong Match", scoring_notes=[])
    ranked = ScreeningPipeline.rank_candidates([candidate_a, candidate_b])
    assert ranked[0].candidate_name == "Alpha"
    assert ranked[1].candidate_name == "Beta"


def test_tie_breaking_uses_skill_then_semantic_then_name_then_filename() -> None:
    first = CandidateResult(candidate_name="Zed", original_filename="z.txt", email="z@example.com", final_score=80.0, semantic_relevance_score=70.0, skill_match_score=90.0, experience_match_score=100.0, education_match_score=100.0, matched_skills=["Python"], missing_skills=[], extracted_skills=["Python"], extracted_experience=3.0, required_experience=2.0, extracted_education=["B.Tech"], required_education=["B.Tech"], recommendation="Strong Match", scoring_notes=[])
    second = CandidateResult(candidate_name="Amy", original_filename="a.txt", email="a@example.com", final_score=80.0, semantic_relevance_score=80.0, skill_match_score=80.0, experience_match_score=100.0, education_match_score=100.0, matched_skills=["Python"], missing_skills=[], extracted_skills=["Python"], extracted_experience=3.0, required_experience=2.0, extracted_education=["B.Tech"], required_education=["B.Tech"], recommendation="Strong Match", scoring_notes=[])
    ranked = ScreeningPipeline.rank_candidates([first, second])
    assert ranked[0].candidate_name == "Zed"
    assert ranked[1].candidate_name == "Amy"


def test_duplicate_candidate_names_are_preserved() -> None:
    first = CandidateResult(candidate_name="Same", original_filename="one.txt", email="one@example.com", final_score=80.0, semantic_relevance_score=80.0, skill_match_score=70.0, experience_match_score=100.0, education_match_score=100.0, matched_skills=["Python"], missing_skills=[], extracted_skills=["Python"], extracted_experience=3.0, required_experience=2.0, extracted_education=["B.Tech"], required_education=["B.Tech"], recommendation="Strong Match", scoring_notes=[])
    second = CandidateResult(candidate_name="Same", original_filename="two.txt", email="two@example.com", final_score=80.0, semantic_relevance_score=80.0, skill_match_score=70.0, experience_match_score=100.0, education_match_score=100.0, matched_skills=["Python"], missing_skills=[], extracted_skills=["Python"], extracted_experience=3.0, required_experience=2.0, extracted_education=["B.Tech"], required_education=["B.Tech"], recommendation="Strong Match", scoring_notes=[])
    ranked = ScreeningPipeline.rank_candidates([first, second])
    assert len(ranked) == 2
    assert {result.original_filename for result in ranked} == {"one.txt", "two.txt"}


def test_all_invalid_resumes_are_reported() -> None:
    pipeline = ScreeningPipeline(job_description="Python", resume_dir=Path("tests/fixtures"))
    results, failures = pipeline.run([Path("tests/fixtures/not-a-resume.xyz")])
    assert results == []
    assert len(failures) == 1
    assert failures[0].reason == "unsupported file type"
