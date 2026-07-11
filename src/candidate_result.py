"""Structured candidate result model for ranking and export."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateResult:
    """Structured output for one screened candidate."""

    candidate_name: str
    original_filename: str
    email: str | None = None
    final_score: float = 0.0
    semantic_relevance_score: float = 0.0
    skill_match_score: float = 0.0
    experience_match_score: float = 0.0
    education_match_score: float = 0.0
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    extracted_skills: list[str] = field(default_factory=list)
    extracted_experience: float | None = None
    required_experience: float | None = None
    extracted_education: list[str] = field(default_factory=list)
    required_education: list[str] = field(default_factory=list)
    recommendation: str = "Low Match"
    strengths: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    explanation: str = ""
    scoring_notes: list[str] = field(default_factory=list)
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "candidate_name": self.candidate_name,
            "original_filename": self.original_filename,
            "email": self.email,
            "final_score": round(self.final_score, 2),
            "semantic_relevance_score": round(self.semantic_relevance_score, 2),
            "skill_match_score": round(self.skill_match_score, 2),
            "experience_match_score": round(self.experience_match_score, 2),
            "education_match_score": round(self.education_match_score, 2),
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "extracted_skills": self.extracted_skills,
            "extracted_experience": self.extracted_experience,
            "required_experience": self.required_experience,
            "extracted_education": self.extracted_education,
            "required_education": self.required_education,
            "recommendation": self.recommendation,
            "strengths": self.strengths,
            "gaps": self.gaps,
            "explanation": self.explanation,
            "scoring_notes": self.scoring_notes,
        }
