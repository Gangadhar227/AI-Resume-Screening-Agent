"""Scoring logic for ranking candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.information_extractor import SKILL_ALIASES, SKILL_ORDER, canonicalize_education, canonicalize_skill, normalize_education, normalize_skills


@dataclass(frozen=True)
class ScoringWeights:
    """Configurable scoring weights that must total 100%."""

    semantic: float = 50.0
    skill: float = 30.0
    experience: float = 15.0
    education: float = 5.0

    def __post_init__(self) -> None:
        for name in ("semantic", "skill", "experience", "education"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)):
                raise ValueError(f"Weight '{name}' must be numeric.")
            if value < 0:
                raise ValueError(f"Weight '{name}' cannot be negative.")
            if value > 100:
                raise ValueError(f"Weight '{name}' cannot exceed 100.")

        total = self.semantic + self.skill + self.experience + self.education
        if abs(total - 100.0) > 1e-9:
            raise ValueError("Scoring weights must total 100%.")

    @property
    def total_weight(self) -> float:
        return self.semantic + self.skill + self.experience + self.education


class RecommendationTier:
    """Centralized recommendation tiers for decision-support scoring."""

    def __init__(self, min_score: float, label: str, description: str) -> None:
        self.min_score = min_score
        self.label = label
        self.description = description

    @classmethod
    def from_score(cls, score: float) -> "RecommendationTier":
        if score >= 80:
            return cls(80.0, "Strong Match", "Recommended for Shortlisting")
        if score >= 65:
            return cls(65.0, "Good Match", "Consider")
        if score >= 50:
            return cls(50.0, "Moderate Match", "Manual Review")
        return cls(0.0, "Low Match", "Low Match")


@dataclass
class ScoreResult:
    """Structured scoring output used for downstream display and export."""

    final_score: float
    semantic_relevance_score: float
    skill_match_score: float
    experience_match_score: float
    education_match_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    candidate_experience: float | None
    required_experience: float | None
    candidate_education: list[str]
    required_education: list[str]
    recommendation: str
    scoring_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_score": round(self.final_score, 2),
            "semantic_relevance_score": round(self.semantic_relevance_score, 2),
            "skill_match_score": round(self.skill_match_score, 2),
            "experience_match_score": round(self.experience_match_score, 2),
            "education_match_score": round(self.education_match_score, 2),
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "candidate_experience": self.candidate_experience,
            "required_experience": self.required_experience,
            "candidate_education": self.candidate_education,
            "required_education": self.required_education,
            "recommendation": self.recommendation,
            "scoring_notes": self.scoring_notes,
        }


def validate_weights(weights: ScoringWeights) -> None:
    """Validate scoring weights before use."""
    if not isinstance(weights, ScoringWeights):
        raise ValueError("weights must be a ScoringWeights instance.")


def _normalize_skills(skills: list[str]) -> list[str]:
    return normalize_skills(skills)


def _canonicalize_skill(skill: str) -> str:
    return canonicalize_skill(skill)


def calculate_skill_match_score(required_skills: list[str], candidate_skills: list[str]) -> tuple[float, list[str], list[str], str | None]:
    """Calculate the skill match score and return matched/missing skills."""
    normalized_required = _normalize_skills(required_skills)
    normalized_candidate = _normalize_skills(candidate_skills)

    if not normalized_required:
        return 50.0, [], [], "The JD did not provide enough identifiable technical-skill requirements for a meaningful skill-match score."

    candidate_lookup = {canonicalize_skill(item).lower() for item in normalized_candidate}
    matched = [skill for skill in normalized_required if skill.lower() in candidate_lookup]
    missing = [skill for skill in normalized_required if skill.lower() not in candidate_lookup]

    score = (len(matched) / len(normalized_required)) * 100.0 if normalized_required else 0.0
    return round(score, 2), matched, missing, None


def extract_job_requirements(job_description: str) -> dict[str, Any]:
    """Extract required skills, experience, and education from the job description."""
    text = (job_description or "").strip()
    required_skills: list[str] = []
    for skill in SKILL_ORDER:
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(skill.lower())}(?![A-Za-z0-9])", text.lower()):
            required_skills.append(skill)

    for alias, canonical in SKILL_ALIASES.items():
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(alias.lower())}(?![A-Za-z0-9])", text.lower()):
            if canonical not in required_skills:
                required_skills.append(canonical)

    required_experience: float | None = None
    # 1. Match range first to avoid matching the upper limit as a single year requirement
    range_match = re.search(r"(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)\s*years?", text, flags=re.IGNORECASE)
    if range_match:
        required_experience = float(range_match.group(1))
    else:
        # 2. Match single year requirements
        single_match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:\+)?\s*years?", text, flags=re.IGNORECASE)
        if single_match:
            required_experience = float(single_match.group(1))

    required_education: list[str] = []
    # Search for Bachelor's and Master's requirements from natural wording and abbreviations
    if re.search(r"\bBachelor(?:'s|’s)?(?:\s+degree)?\b|\bB\.Tech\b|\bBachelor of Technology\b|\bB\.E\.?\b|\bBachelor of Engineering\b|\bB\.Sc\b|\bBCA\b|\bBS\b|\bB\.S\b", text, flags=re.IGNORECASE):
        required_education.append("Bachelor")
    if re.search(r"\bMaster(?:'s|’s)?(?:\s+degree)?\b|\bM\.Tech\b|\bMaster of Technology\b|\bM\.Sc\b|\bMBA\b|\bMCA\b|\bMS\b|\bM\.S\b", text, flags=re.IGNORECASE):
        required_education.append("Master")

    return {
        "required_skills": _normalize_skills(required_skills),
        "required_experience": required_experience,
        "required_education": normalize_education(required_education, prefer_degree_labels=True) if required_education else [],
    }


def calculate_candidate_score(
    semantic_score: float,
    required_skills: list[str],
    candidate_skills: list[str],
    candidate_experience: float | None,
    required_experience: float | None,
    candidate_education: list[str],
    required_education: list[str],
    weights: ScoringWeights | None = None,
) -> ScoreResult:
    """Calculate an explainable weighted score for one candidate."""
    weights = weights or ScoringWeights()
    validate_weights(weights)

    semantic_relevance_score = max(0.0, min(100.0, float(semantic_score)))
    skill_match_score, matched_skills, missing_skills, skill_note = calculate_skill_match_score(required_skills, candidate_skills)
    skill_note = skill_note or ("" if matched_skills or missing_skills or required_skills else "The JD did not provide enough identifiable technical-skill requirements for a meaningful skill-match score.")

    if required_experience is None:
        experience_match_score = 50.0
        experience_note = "No reliable minimum experience requirement was found in the JD; using a neutral score."
    elif candidate_experience is None:
        experience_match_score = 50.0
        experience_note = "Candidate experience is unknown; using a neutral score."
    elif required_experience <= 0:
        experience_match_score = 100.0
        experience_note = "The JD specified a zero-year requirement; assigning a neutral maximum score."
    elif candidate_experience >= required_experience:
        experience_match_score = 100.0
        experience_note = "Candidate meets or exceeds the minimum stated experience requirement."
    else:
        experience_match_score = max(0.0, min(100.0, (candidate_experience / required_experience) * 100.0))
        experience_note = "Candidate experience is below the minimum stated requirement."

    if not required_education:
        education_match_score = 50.0
        education_note = "No identifiable education requirement found in the JD; using a neutral score."
    elif not candidate_education:
        education_match_score = 50.0
        education_note = "Candidate education is unknown; using a neutral score."
    else:
        candidate_education_norm = [canonicalize_education(item, prefer_degree_labels=True) for item in candidate_education if item and item.strip()]
        required_education_norm = [canonicalize_education(item, prefer_degree_labels=True) for item in required_education if item and item.strip()]
        if any(item in candidate_education_norm for item in required_education_norm):
            education_match_score = 100.0
            education_note = "Candidate education matches an identified requirement."
        else:
            education_match_score = 0.0
            education_note = "Candidate education does not match the identified requirement."

    final_score = (
        semantic_relevance_score * (weights.semantic / 100.0)
        + skill_match_score * (weights.skill / 100.0)
        + experience_match_score * (weights.experience / 100.0)
        + education_match_score * (weights.education / 100.0)
    )
    final_score = max(0.0, min(100.0, final_score))
    tier = RecommendationTier.from_score(final_score)
    notes = [note for note in [skill_note, experience_note, education_note] if note]
    return ScoreResult(
        final_score=round(final_score, 2),
        semantic_relevance_score=round(semantic_relevance_score, 2),
        skill_match_score=round(skill_match_score, 2),
        experience_match_score=round(experience_match_score, 2),
        education_match_score=round(education_match_score, 2),
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        candidate_experience=candidate_experience,
        required_experience=required_experience,
        candidate_education=candidate_education,
        required_education=required_education,
        recommendation=tier.label,
        scoring_notes=notes,
    )


def calculate_scores(*args, **kwargs) -> dict:
    """Backward-compatible wrapper for weighted candidate scoring."""
    return calculate_candidate_score(*args, **kwargs).to_dict()
