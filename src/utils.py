"""Utility helpers for the AI Resume Screening Agent."""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any, Iterable

from src.candidate_result import CandidateResult


def get_project_root() -> Path:
    """Return the repository root path."""
    return Path(__file__).resolve().parents[1]


def _serialize_list(values: Iterable[Any]) -> str:
    return "; ".join(str(value) for value in values if value)


def export_candidates_to_csv(candidates: list[CandidateResult], output_path: str | Path | None = None) -> str:
    """Return CSV content for candidate results and optionally write it to disk."""
    fieldnames = [
        "Rank",
        "Candidate Name",
        "File Name",
        "Email",
        "Final Score",
        "Semantic Relevance Score",
        "Skill Match Score",
        "Experience Match Score",
        "Education Match Score",
        "Matched Skills",
        "Missing Skills",
        "Extracted Skills",
        "Extracted Experience",
        "Required Experience",
        "Extracted Education",
        "Required Education",
        "Recommendation",
        "Strengths",
        "Gaps",
        "Explanation",
    ]
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "Rank": candidate.rank or 0,
                "Candidate Name": candidate.candidate_name,
                "File Name": candidate.original_filename,
                "Email": candidate.email or "",
                "Final Score": round(candidate.final_score, 2),
                "Semantic Relevance Score": round(candidate.semantic_relevance_score, 2),
                "Skill Match Score": round(candidate.skill_match_score, 2),
                "Experience Match Score": round(candidate.experience_match_score, 2),
                "Education Match Score": round(candidate.education_match_score, 2),
                "Matched Skills": _serialize_list(candidate.matched_skills),
                "Missing Skills": _serialize_list(candidate.missing_skills),
                "Extracted Skills": _serialize_list(candidate.extracted_skills),
                "Extracted Experience": candidate.extracted_experience,
                "Required Experience": candidate.required_experience,
                "Extracted Education": _serialize_list(candidate.extracted_education),
                "Required Education": _serialize_list(candidate.required_education),
                "Recommendation": candidate.recommendation,
                "Strengths": _serialize_list(candidate.strengths),
                "Gaps": _serialize_list(candidate.gaps),
                "Explanation": candidate.explanation,
            }
        )

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_text = buffer.getvalue()

    if output_path is not None:
        path = Path(output_path)
        path.write_text(csv_text, encoding="utf-8")

    return csv_text


def export_candidates_to_json(candidates: list[CandidateResult], output_path: str | Path | None = None) -> str:
    """Return JSON content for candidate results and optionally write it to disk."""
    payload = [candidate.to_dict() for candidate in candidates]
    json_text = json.dumps(payload, indent=2, ensure_ascii=False)

    if output_path is not None:
        Path(output_path).write_text(json_text, encoding="utf-8")

    return json_text
