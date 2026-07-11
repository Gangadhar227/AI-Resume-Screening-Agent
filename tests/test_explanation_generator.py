from __future__ import annotations

from src.candidate_result import CandidateResult
from src.explanation_generator import generate_explanation


def test_strong_candidate_explanation() -> None:
    candidate = CandidateResult(
        candidate_name="Ananya Rao",
        original_filename="ananya.pdf",
        email="ananya@example.com",
        final_score=84.25,
        semantic_relevance_score=90.0,
        skill_match_score=80.0,
        experience_match_score=100.0,
        education_match_score=100.0,
        matched_skills=["Python", "NLP"],
        missing_skills=["Docker", "AWS"],
        extracted_skills=["Python", "NLP", "SQL"],
        extracted_experience=4.0,
        required_experience=3.0,
        extracted_education=["B.Tech"],
        required_education=["B.Tech"],
        recommendation="Strong Match",
        scoring_notes=["Candidate meets the experience requirement."],
    )

    explanation = generate_explanation(candidate)

    assert "84.25" in explanation
    assert "strong semantic alignment" in explanation.lower()
    assert "2 of 2" in explanation or "2 of 2" in explanation.lower()
    assert "recommended for human shortlisting review" in explanation.lower()
    assert "guaranteed" not in explanation.lower()
    assert "hired" not in explanation.lower()
    assert "probability" not in explanation.lower()


def test_partial_match_explanation_mentions_missing_skills() -> None:
    candidate = CandidateResult(
        candidate_name="Priya",
        original_filename="priya.docx",
        email="priya@example.com",
        final_score=56.5,
        semantic_relevance_score=60.0,
        skill_match_score=33.33,
        experience_match_score=50.0,
        education_match_score=50.0,
        matched_skills=["Python"],
        missing_skills=["Docker", "AWS"],
        extracted_skills=["Python"],
        extracted_experience=1.0,
        required_experience=3.0,
        extracted_education=[],
        required_education=["B.Tech"],
        recommendation="Moderate Match",
        scoring_notes=["Candidate experience is below the requirement."],
    )

    explanation = generate_explanation(candidate)

    assert "missing skills" in explanation.lower()
    assert "docker" in explanation.lower()
    assert "aws" in explanation.lower()


def test_missing_experience_and_education_are_honest() -> None:
    candidate = CandidateResult(
        candidate_name="Ravi",
        original_filename="ravi.txt",
        email="ravi@example.com",
        final_score=42.0,
        semantic_relevance_score=40.0,
        skill_match_score=25.0,
        experience_match_score=50.0,
        education_match_score=50.0,
        matched_skills=[],
        missing_skills=["Python"],
        extracted_skills=[],
        extracted_experience=None,
        required_experience=3.0,
        extracted_education=[],
        required_education=["M.Tech"],
        recommendation="Low Match",
        scoring_notes=["Candidate experience is unknown."],
    )

    explanation = generate_explanation(candidate)

    assert "experience could not be identified" in explanation.lower()
    assert "education requirement could not be confirmed" in explanation.lower()


def test_explanation_does_not_repeat_missing_skills() -> None:
    candidate = CandidateResult(
        candidate_name="Mina",
        original_filename="mina.txt",
        email="mina@example.com",
        final_score=42.0,
        semantic_relevance_score=40.0,
        skill_match_score=25.0,
        experience_match_score=50.0,
        education_match_score=50.0,
        matched_skills=[],
        missing_skills=["Docker", "AWS"],
        extracted_skills=[],
        extracted_experience=None,
        required_experience=3.0,
        extracted_education=[],
        required_education=["B.Tech"],
        recommendation="Low Match",
        scoring_notes=["Candidate experience is unknown."],
    )

    explanation = generate_explanation(candidate)

    assert explanation.lower().count("missing skills") == 1
    assert "review the complete resume before making any decision" in explanation.lower()
