from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from src.candidate_result import CandidateResult
from src.utils import export_candidates_to_csv, export_candidates_to_json


def test_csv_generation_contains_all_candidates_and_columns() -> None:
    candidates = [
        CandidateResult(
            candidate_name="Asha",
            original_filename="asha.pdf",
            email="asha@example.com",
            final_score=84.5,
            semantic_relevance_score=90.0,
            skill_match_score=80.0,
            experience_match_score=100.0,
            education_match_score=100.0,
            matched_skills=["Python"],
            missing_skills=["Docker"],
            extracted_skills=["Python", "SQL"],
            extracted_experience=3.0,
            required_experience=3.0,
            extracted_education=["B.Tech"],
            required_education=["B.Tech"],
            recommendation="Strong Match",
            strengths=["Strong semantic alignment"],
            gaps=["Missing Docker"],
            explanation="Example explanation",
            scoring_notes=["Note"],
        )
    ]

    csv_text = export_candidates_to_csv(candidates)
    assert "Rank" in csv_text
    assert "Candidate Name" in csv_text
    assert "Final Score" in csv_text
    assert "Asha" in csv_text
    assert "Python" in csv_text


def test_csv_file_writing_uses_utf8(tmp_path: Path) -> None:
    output_path = tmp_path / "results.csv"
    export_candidates_to_csv([], output_path=output_path)
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8")


def test_json_generation_contains_arrays_and_is_valid(tmp_path: Path) -> None:
    candidates = [
        CandidateResult(
            candidate_name="Asha",
            original_filename="asha.pdf",
            email="asha@example.com",
            final_score=84.5,
            semantic_relevance_score=90.0,
            skill_match_score=80.0,
            experience_match_score=100.0,
            education_match_score=100.0,
            matched_skills=["Python"],
            missing_skills=["Docker"],
            extracted_skills=["Python", "SQL"],
            extracted_experience=3.0,
            required_experience=3.0,
            extracted_education=["B.Tech"],
            required_education=["B.Tech"],
            recommendation="Strong Match",
            strengths=["Strong semantic alignment"],
            gaps=["Missing Docker"],
            explanation="Example explanation",
            scoring_notes=["Note"],
        )
    ]

    payload = export_candidates_to_json(candidates)
    parsed = json.loads(payload)
    assert parsed[0]["candidate_name"] == "Asha"
    assert parsed[0]["matched_skills"] == ["Python"]
    assert parsed[0]["strengths"] == ["Strong semantic alignment"]

    output_path = tmp_path / "results.json"
    export_candidates_to_json(candidates, output_path=output_path)
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8"))
