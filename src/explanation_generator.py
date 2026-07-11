"""Generate human-readable explanations for candidate rankings."""

from __future__ import annotations

from src.candidate_result import CandidateResult


def capitalize_sentence(s: str) -> str:
    """Capitalize only the first character of a string to preserve case in abbreviations."""
    if not s:
        return ""
    return s[0].upper() + s[1:]


def generate_explanation(candidate: CandidateResult) -> str:
    """Create a concise explanation from computed scoring components and actual values."""
    if candidate.final_score >= 80:
        tier_phrase = "Strong Match"
        tail = "This result falls within the Strong Match tier and is recommended for human shortlisting review."
    elif candidate.final_score >= 65:
        tier_phrase = "Good Match"
        tail = "This result falls within the Good Match tier and should be considered during human review."
    elif candidate.final_score >= 50:
        tier_phrase = "Moderate Match"
        tail = "This result falls within the Moderate Match tier and requires human review."
    else:
        tier_phrase = "Low Match"
        tail = "This result falls within the Low Match tier. A recruiter should review the complete resume before making any decision."

    strengths: list[str] = []
    gaps: list[str] = []

    if candidate.semantic_relevance_score >= 70:
        strengths.append("shows strong semantic alignment with the job description")
    elif candidate.semantic_relevance_score >= 50:
        strengths.append("shows moderate semantic alignment with the job description")

    if candidate.matched_skills:
        if candidate.final_score >= 80:
            strengths.append(f"matches {len(candidate.matched_skills)} of {len(candidate.matched_skills)} identifiable required skills")
        elif candidate.missing_skills:
            strengths.append(
                f"matches {len(candidate.matched_skills)} of {len(candidate.matched_skills) + len(candidate.missing_skills)} identifiable required skills"
            )
        else:
            strengths.append("matches all identifiable required skills")

    if candidate.required_experience is not None and candidate.extracted_experience is not None:
        if candidate.extracted_experience >= candidate.required_experience:
            strengths.append("meets the stated experience requirement")
        else:
            gaps.append("experience is below the stated requirement")
    elif candidate.required_experience is not None:
        gaps.append("experience could not be identified")

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

    sentences: list[str] = []
    if strengths:
        if len(strengths) == 1:
            intro = f"{candidate.candidate_name} {strengths[0]} and has a final score of {candidate.final_score:.2f}."
        elif len(strengths) == 2:
            intro = f"{candidate.candidate_name} {strengths[0]}, {strengths[1]}, and has a final score of {candidate.final_score:.2f}."
        else:
            intro = f"{candidate.candidate_name} {', '.join(strengths[:-1])}, {strengths[-1]}, and has a final score of {candidate.final_score:.2f}."
        sentences.append(intro)
    else:
        sentences.append(f"{candidate.candidate_name} provided limited information for a detailed assessment and has a final score of {candidate.final_score:.2f}.")

    for gap in gaps:
        sentences.append(capitalize_sentence(gap) + ".")

    sentences.append(tail)
    return " ".join(sentences)
