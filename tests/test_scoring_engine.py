from __future__ import annotations

import pytest

from src.scoring_engine import (
    RecommendationTier,
    ScoringWeights,
    ScoreResult,
    calculate_candidate_score,
    calculate_skill_match_score,
    extract_job_requirements,
    validate_weights,
)


def test_default_weights_total_100() -> None:
    weights = ScoringWeights()
    assert weights.total_weight == pytest.approx(100.0)


def test_valid_custom_weights() -> None:
    weights = ScoringWeights(semantic=40.0, skill=30.0, experience=20.0, education=10.0)
    assert weights.semantic == 40.0
    assert weights.total_weight == pytest.approx(100.0)


def test_invalid_weight_total_raises() -> None:
    with pytest.raises(ValueError):
        ScoringWeights(semantic=0.5, skill=0.2, experience=0.1, education=0.1)


def test_negative_weights_raise() -> None:
    with pytest.raises(ValueError):
        ScoringWeights(semantic=-0.1, skill=0.6, experience=0.3, education=0.2)


def test_weight_above_100_raises() -> None:
    with pytest.raises(ValueError):
        ScoringWeights(semantic=101, skill=0, experience=0, education=0)


def test_skill_match_calculation_is_case_insensitive() -> None:
    score, matched, missing, note = calculate_skill_match_score(
        required_skills=["Python", "NLP"],
        candidate_skills=["python", "sql"],
    )
    assert score == 50.0
    assert matched == ["Python"]
    assert missing == ["NLP"]
    assert note is None


def test_no_jd_skills_gives_neutral_score() -> None:
    score, matched, missing, note = calculate_skill_match_score(required_skills=[], candidate_skills=["Python"])
    assert score == 50.0
    assert matched == []
    assert missing == []
    assert "JD" in note


def test_full_skill_match() -> None:
    score, matched, missing, note = calculate_skill_match_score(required_skills=["Python", "SQL"], candidate_skills=["Python", "SQL"])
    assert score == 100.0
    assert matched == ["Python", "SQL"]
    assert missing == []
    assert note is None


def test_partial_skill_match() -> None:
    score, matched, missing, note = calculate_skill_match_score(required_skills=["Python", "SQL", "NLP"], candidate_skills=["Python"])
    assert score == 33.33
    assert matched == ["Python"]
    assert missing == ["SQL", "NLP"]
    assert note is None


def test_extract_job_requirements() -> None:
    requirements = extract_job_requirements("We need Python, NLP, and 3 years of experience. Education: B.Tech or M.Tech.")
    assert "Python" in requirements["required_skills"]
    assert "Natural Language Processing" in requirements["required_skills"]
    assert requirements["required_experience"] == 3.0
    assert requirements["required_education"] == ["Bachelor", "Master"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0–2 years", 0.0),
        ("1–3 years", 1.0),
        ("minimum 2 years", 2.0),
        ("2+ years", 2.0),
    ],
)
def test_extract_job_requirements_uses_minimum_range_value(text: str, expected: float) -> None:
    requirements = extract_job_requirements(text)
    assert requirements["required_experience"] == expected


def test_experience_requirement_missing() -> None:
    requirements = extract_job_requirements("We need Python skills only")
    assert requirements["required_experience"] is None


def test_candidate_meets_experience_requirement() -> None:
    result = calculate_candidate_score(
        semantic_score=80,
        required_skills=["Python"],
        candidate_skills=["Python"],
        candidate_experience=4.0,
        required_experience=3.0,
        candidate_education=["B.Tech"],
        required_education=["B.Tech"],
    )
    assert result.final_score >= 80
    assert result.experience_match_score == 100.0


def test_candidate_partially_meets_experience_requirement() -> None:
    result = calculate_candidate_score(
        semantic_score=60,
        required_skills=["Python"],
        candidate_skills=["Python"],
        candidate_experience=1.0,
        required_experience=3.0,
        candidate_education=["B.Tech"],
        required_education=["B.Tech"],
    )
    assert result.experience_match_score == 33.33


def test_candidate_experience_unknown() -> None:
    result = calculate_candidate_score(
        semantic_score=60,
        required_skills=["Python"],
        candidate_skills=["Python"],
        candidate_experience=None,
        required_experience=3.0,
        candidate_education=["B.Tech"],
        required_education=["B.Tech"],
    )
    assert result.experience_match_score == 50.0


def test_education_match() -> None:
    result = calculate_candidate_score(
        semantic_score=60,
        required_skills=["Python"],
        candidate_skills=["Python"],
        candidate_experience=3.0,
        required_experience=3.0,
        candidate_education=["B.Tech"],
        required_education=["B.Tech"],
    )
    assert result.education_match_score == 100.0


def test_education_mismatch() -> None:
    result = calculate_candidate_score(
        semantic_score=60,
        required_skills=["Python"],
        candidate_skills=["Python"],
        candidate_experience=3.0,
        required_experience=3.0,
        candidate_education=["M.Sc"],
        required_education=["B.Tech"],
    )
    assert result.education_match_score == 0.0


def test_missing_jd_education_requirement() -> None:
    result = calculate_candidate_score(
        semantic_score=60,
        required_skills=["Python"],
        candidate_skills=["Python"],
        candidate_experience=3.0,
        required_experience=3.0,
        candidate_education=["B.Tech"],
        required_education=[],
    )
    assert result.education_match_score == 50.0


def test_weighted_final_score_is_calculated() -> None:
    result = calculate_candidate_score(
        semantic_score=80,
        required_skills=["Python"],
        candidate_skills=["Python"],
        candidate_experience=4.0,
        required_experience=3.0,
        candidate_education=["B.Tech"],
        required_education=["B.Tech"],
    )
    assert result.final_score == pytest.approx(90.0)


def test_score_clamping() -> None:
    result = calculate_candidate_score(
        semantic_score=100,
        required_skills=["Python"],
        candidate_skills=[],
        candidate_experience=10.0,
        required_experience=1.0,
        candidate_education=["B.Tech"],
        required_education=["B.Tech"],
    )
    assert result.final_score <= 100


def test_recommendation_boundaries() -> None:
    assert RecommendationTier.from_score(0).label == "Low Match"
    assert RecommendationTier.from_score(49.99).label == "Low Match"
    assert RecommendationTier.from_score(50).label == "Moderate Match"
    assert RecommendationTier.from_score(64.99).label == "Moderate Match"
    assert RecommendationTier.from_score(65).label == "Good Match"
    assert RecommendationTier.from_score(79.99).label == "Good Match"
    assert RecommendationTier.from_score(80).label == "Strong Match"
    assert RecommendationTier.from_score(100).label == "Strong Match"


def test_structured_result_conversion() -> None:
    result = calculate_candidate_score(
        semantic_score=80,
        required_skills=["Python"],
        candidate_skills=["Python"],
        candidate_experience=4.0,
        required_experience=3.0,
        candidate_education=["B.Tech"],
        required_education=["B.Tech"],
    )
    payload = result.to_dict()
    assert payload["final_score"] == pytest.approx(result.final_score)
    assert payload["recommendation"] == result.recommendation
