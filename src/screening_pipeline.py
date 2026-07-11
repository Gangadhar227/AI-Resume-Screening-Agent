"""Reusable screening pipeline for ranking candidates from resumes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.candidate_result import CandidateResult
from src.explanation_generator import generate_explanation
from src.information_extractor import extract_candidate_information
from src.resume_parser import extract_text_from_file
from src.scoring_engine import ScoringWeights, calculate_candidate_score, extract_job_requirements
from src.similarity_engine import SemanticSimilarityEngine


@dataclass(frozen=True)
class ScreeningFailure:
    """Container for a resume that could not be processed."""

    filename: str
    reason: str
    details: str | None = None


class ScreeningPipeline:
    """Coordinate parsing, scoring, explanation generation, and ranking."""

    def __init__(self, job_description: str, resume_dir: Path | None = None, semantic_engine: SemanticSimilarityEngine | None = None) -> None:
        self.job_description = job_description
        self.resume_dir = resume_dir or Path("data")
        self.semantic_engine = semantic_engine
        self.job_requirements = extract_job_requirements(job_description)
        self.weights = ScoringWeights()

    @staticmethod
    def rank_candidates(candidates: list[CandidateResult]) -> list[CandidateResult]:
        """Rank candidates by final score with deterministic tie-breaking."""
        ranked = sorted(
            candidates,
            key=lambda item: (
                -item.final_score,
                -item.skill_match_score,
                -item.semantic_relevance_score,
                item.candidate_name.lower(),
                item.original_filename.lower(),
            ),
        )
        for index, candidate in enumerate(ranked, start=1):
            candidate.rank = index
        return ranked

    def _process_single_resume(self, resume_path: Path) -> tuple[CandidateResult | None, ScreeningFailure | None]:
        try:
            text = extract_text_from_file(resume_path)
        except Exception as exc:
            return None, ScreeningFailure(filename=resume_path.name, reason="processing error", details=str(exc))

        if not text or not text.strip():
            return None, ScreeningFailure(filename=resume_path.name, reason="empty content", details="The parsed resume was empty.")

        try:
            extracted = extract_candidate_information(text, filename=resume_path.name)
        except Exception as exc:
            return None, ScreeningFailure(filename=resume_path.name, reason="extraction error", details=str(exc))

        return self._build_candidate_from_extracted(resume_path, text, extracted)

    def _build_candidate_from_extracted(self, resume_path: Path, text: str, extracted: dict[str, Any], similarity_score: float | None = None) -> tuple[CandidateResult | None, ScreeningFailure | None]:
        if similarity_score is None:
            try:
                similarity_score = self.semantic_engine.compute_similarity(self.job_description, text)
            except Exception as exc:
                return None, ScreeningFailure(filename=resume_path.name, reason="semantic model error", details=str(exc))

        required_skills = self.job_requirements.get("required_skills", [])
        required_experience = self.job_requirements.get("required_experience")
        required_education = self.job_requirements.get("required_education", [])

        score_result = calculate_candidate_score(
            semantic_score=similarity_score,
            required_skills=required_skills,
            candidate_skills=extracted.get("skills", []),
            candidate_experience=extracted.get("experience_years"),
            required_experience=required_experience,
            candidate_education=extracted.get("education", []),
            required_education=required_education,
            weights=self.weights,
        )

        candidate = CandidateResult(
            candidate_name=extracted.get("name") or resume_path.stem,
            original_filename=resume_path.name,
            email=extracted.get("email"),
            final_score=score_result.final_score,
            semantic_relevance_score=score_result.semantic_relevance_score,
            skill_match_score=score_result.skill_match_score,
            experience_match_score=score_result.experience_match_score,
            education_match_score=score_result.education_match_score,
            matched_skills=score_result.matched_skills,
            missing_skills=score_result.missing_skills,
            extracted_skills=extracted.get("skills", []),
            extracted_experience=extracted.get("experience_years"),
            required_experience=required_experience,
            extracted_education=extracted.get("education", []),
            required_education=required_education,
            recommendation=score_result.recommendation,
            scoring_notes=score_result.scoring_notes,
        )
        candidate.strengths = []
        candidate.gaps = []

        if candidate.semantic_relevance_score >= 70:
            candidate.strengths.append("Strong semantic alignment")
        elif candidate.semantic_relevance_score >= 50:
            candidate.strengths.append("Moderate semantic alignment")

        if candidate.skill_match_score >= 75.0:
            candidate.strengths.append("High required-skill coverage")
        elif candidate.skill_match_score < 50.0 and candidate.missing_skills:
            candidate.gaps.append("Missing specific required skills")

        if candidate.required_experience is not None:
            if candidate.experience_match_score == 100.0:
                candidate.strengths.append("Meets the stated experience requirement")
            elif candidate.extracted_experience is None:
                candidate.gaps.append("Experience could not be identified")
            else:
                candidate.gaps.append("Experience below requirement")

        if candidate.required_education:
            if candidate.education_match_score == 100.0:
                candidate.strengths.append("Matches the identified education requirement")
            else:
                candidate.gaps.append("Education requirement could not be confirmed")

        candidate.explanation = generate_explanation(candidate)
        return candidate, None

    def run(self, resume_paths: list[Path] | None = None) -> tuple[list[CandidateResult], list[ScreeningFailure]]:
        """Process a collection of resumes and return ranked candidates plus failures."""
        path_list = list(resume_paths or [])
        if not path_list:
            return [], []

        results: list[CandidateResult] = []
        failures: list[ScreeningFailure] = []
        processed_inputs: list[tuple[Path, str, dict[str, Any]]] = []

        for path in path_list:
            suffix = path.suffix.lower()
            if suffix not in {".pdf", ".docx", ".txt", ".text"}:
                failures.append(ScreeningFailure(filename=path.name, reason="unsupported file type", details="Only PDF, DOCX, and TXT files are supported."))
                continue
            if not path.exists():
                failures.append(ScreeningFailure(filename=path.name, reason="file not found", details="The requested resume file could not be found."))
                continue
            try:
                text = extract_text_from_file(path)
            except Exception as exc:
                failures.append(ScreeningFailure(filename=path.name, reason="processing error", details=str(exc)))
                continue
            if not text or not text.strip():
                failures.append(ScreeningFailure(filename=path.name, reason="empty content", details="The parsed resume was empty."))
                continue
            try:
                extracted = extract_candidate_information(text, filename=path.name)
            except Exception as exc:
                failures.append(ScreeningFailure(filename=path.name, reason="extraction error", details=str(exc)))
                continue
            processed_inputs.append((path, text, extracted))

        if processed_inputs:
            if self.semantic_engine is None:
                self.semantic_engine = SemanticSimilarityEngine()
            texts = [item[1] for item in processed_inputs]
            try:
                similarity_scores = self.semantic_engine.compute_similarity_batch(self.job_description, texts)
            except Exception as exc:
                for path, _, _ in processed_inputs:
                    failures.append(ScreeningFailure(filename=path.name, reason="semantic model error", details=str(exc)))
                return [], failures

            for (path, text, extracted), similarity_score in zip(processed_inputs, similarity_scores):
                candidate, failure = self._build_candidate_from_extracted(path, text, extracted, similarity_score=similarity_score)
                if failure is not None:
                    failures.append(failure)
                    continue
                if candidate is not None:
                    candidate.explanation = generate_explanation(candidate)
                    results.append(candidate)

        ranked = self.rank_candidates(results)
        return ranked, failures
