"""Generate human-readable explanations for candidate rankings."""

from __future__ import annotations

from src.candidate_result import CandidateResult


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
        strengths.append("strong semantic alignment with the job description")
    elif candidate.semantic_relevance_score >= 50:
        strengths.append("moderate semantic alignment with the job description")

    if candidate.matched_skills:
        if candidate.final_score >= 80:
            strengths.append(f"matched {len(candidate.matched_skills)} of {len(candidate.matched_skills)} identifiable required skills")
        elif candidate.missing_skills:
            strengths.append(
                f"matched {len(candidate.matched_skills)} of {len(candidate.matched_skills) + len(candidate.missing_skills)} identifiable required skills"
            )
        else:
            strengths.append("matched all identifiable required skills")

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
        sentences.append(f"{candidate.candidate_name} shows {', '.join(strengths[0:2])} and has a final score of {candidate.final_score:.2f}.")
    else:
        sentences.append(f"{candidate.candidate_name} provided limited information for a detailed assessment and has a final score of {candidate.final_score:.2f}.")

    for gap in gaps:
        sentences.append(gap.capitalize() + ".")

    if candidate.missing_skills:
        sentence = f"Missing skills include {', '.join(candidate.missing_skills)}."
        if sentence not in sentences:
            sentences.append(sentence)

    sentences.append(tail)
    return " ".join(sentences)
