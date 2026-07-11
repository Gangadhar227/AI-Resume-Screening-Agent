"""Scoring logic for ranking candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.information_extractor import SKILL_ALIASES, SKILL_ORDER


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
    seen: set[str] = set()
    ordered: list[str] = []
    for skill in skills:
        if not skill:
            continue
        canonical = _canonicalize_skill(skill)
        if canonical and canonical not in seen:
            seen.add(canonical)
            ordered.append(canonical)
    return ordered


def _canonicalize_skill(skill: str) -> str:
    if not skill:
        return ""
    cleaned = skill.strip()
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    if lowered in SKILL_ALIASES:
        canonical = SKILL_ALIASES[lowered]
        if lowered in {"nlp", "ml", "rag", "llm", "llms", "genai", "k8s", "js"}:
            return lowered.upper() if lowered in {"nlp", "ml", "llm", "llms", "genai", "k8s", "js"} else canonical
        return canonical
    for alias, canonical in SKILL_ALIASES.items():
        if alias.lower() == lowered:
            if alias in {"nlp", "ml", "rag", "llm", "llms", "genai", "k8s", "js"}:
                return alias.upper() if alias in {"nlp", "ml", "llm", "llms", "genai", "k8s", "js"} else canonical
            return canonical
    if cleaned in SKILL_ORDER:
        return cleaned
    return cleaned


def calculate_skill_match_score(required_skills: list[str], candidate_skills: list[str]) -> tuple[float, list[str], list[str], str | None]:
    """Calculate the skill match score and return matched/missing skills."""
    normalized_required = _normalize_skills(required_skills)
    normalized_candidate = _normalize_skills(candidate_skills)

    if not normalized_required:
        return 50.0, [], [], "The JD did not provide enough identifiable technical-skill requirements for a meaningful skill-match score."

    candidate_lookup = {item.lower() for item in normalized_candidate}
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
            if alias in {"ml", "nlp", "rag", "llm", "llms", "genai", "k8s", "js", "c", "c++"}:
                if alias not in required_skills:
                    required_skills.append(alias.upper() if alias in {"ml", "nlp", "llm", "llms", "genai", "k8s", "js"} else alias)
            elif canonical not in required_skills:
                required_skills.append(canonical)

    if "NLP" not in required_skills and "Natural Language Processing" in required_skills:
        required_skills.remove("Natural Language Processing")
        required_skills.append("NLP")
    if "ML" not in required_skills and "Machine Learning" in required_skills:
        required_skills.remove("Machine Learning")
        required_skills.append("ML")

    experience_patterns = [
        r"(?:minimum|at least|atleast|over|around|about)?\s*(\d+(?:\.\d+)?)\s*\+?\s*years?\s*(?:of\s+)?experience",
        r"(?:minimum|at least|atleast|over|around|about)?\s*(\d+(?:\.\d+)?)\s*\+?\s*years?",
        r"(\d+(?:\.\d+)?)\s*[–-]\s*(\d+(?:\.\d+)?)\s*years?\s*(?:of\s+)?experience",
    ]
    required_experience: float | None = None
    for pattern in experience_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            if len(match.groups()) == 2:
                required_experience = float(match.group(1))
            else:
                required_experience = float(match.group(1))
            break

    required_education: list[str] = []
    education_patterns = [
        (r"\bB\.Tech\b", "B.Tech"),
        (r"\bBachelor of Technology\b", "Bachelor of Technology"),
        (r"\bB\.E\.?\b", "B.E"),
        (r"\bBachelor of Engineering\b", "Bachelor of Engineering"),
        (r"\bB\.Sc\b", "B.Sc"),
        (r"\bBCA\b", "BCA"),
        (r"\bMCA\b", "MCA"),
        (r"\bM\.Tech\b", "M.Tech"),
        (r"\bMaster of Technology\b", "Master of Technology"),
        (r"\bM\.Sc\b", "M.Sc"),
        (r"\bMBA\b", "MBA"),
        (r"\bPhD\b", "PhD"),
    ]
    for pattern, label in education_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            required_education.append(label)
    return {
        "required_skills": _normalize_skills(required_skills),
        "required_experience": required_experience,
        "required_education": required_education,
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
        candidate_education_norm = [item.strip() for item in candidate_education if item and item.strip()]
        required_education_norm = [item.strip() for item in required_education if item and item.strip()]
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
