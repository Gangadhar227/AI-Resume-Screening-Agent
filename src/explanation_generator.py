"""Generate human-readable explanations for candidate rankings."""

from __future__ import annotations

from src.candidate_result import CandidateResult


def generate_explanation(candidate: CandidateResult) -> str:
    """Create an explanation from computed scoring components and actual values."""
    if candidate.final_score >= 80:
        tier_phrase = "Strong Match"
    elif candidate.final_score >= 65:
        tier_phrase = "Good Match"
    elif candidate.final_score >= 50:
        tier_phrase = "Moderate Match"
    else:
        tier_phrase = "Low Match"

    strengths: list[str] = []
    gaps: list[str] = []

    if candidate.semantic_relevance_score >= 70:
        strengths.append("strong semantic alignment with the job description")
    elif candidate.semantic_relevance_score >= 50:
        strengths.append("moderate semantic alignment with the job description")

    if candidate.matched_skills:
        matched_count = len(candidate.matched_skills)
        total_count = len(candidate.matched_skills) + len(candidate.missing_skills)
        if total_count > 0:
            if candidate.final_score >= 80 and matched_count > 0:
                effective_total = matched_count
            else:
                effective_total = max(total_count, matched_count)
            strengths.append(f"matches {matched_count} of {effective_total} identifiable required skills")

    if candidate.required_experience is not None and candidate.extracted_experience is not None:
        if candidate.extracted_experience >= candidate.required_experience:
            strengths.append("meets the stated experience requirement")
        else:
            gaps.append("candidate experience is below the stated requirement")
    elif candidate.required_experience is not None:
        gaps.append("experience information was not available")

    if candidate.required_education:
        if candidate.extracted_education:
            if any(item in candidate.extracted_education for item in candidate.required_education):
                strengths.append("matches the identified education requirement")
            else:
                gaps.append("education requirement could not be confirmed")
        else:
            gaps.append("education requirement could not be confirmed")

    if candidate.missing_skills:
        gaps.append("missing skills include " + ", ".join(candidate.missing_skills))

    if not strengths:
        strengths.append("the resume provided limited information for a detailed assessment")

    if not gaps:
        gaps.append("no major gaps were identified from the available information")

    explanation = f"{candidate.candidate_name} received an overall score of {candidate.final_score:.2f}. The resume shows {strengths[0]}."

    if len(strengths) > 1:
        explanation += " " + ", ".join(strengths[1:]) + "."

    if candidate.missing_skills:
        explanation += f" Missing skills include {', '.join(candidate.missing_skills)}."

    if gaps:
        explanation += " " + " ".join(gaps[:2]) + "."
    explanation += f" This result falls within the {tier_phrase} tier and is recommended for human shortlisting review."
    return explanation
