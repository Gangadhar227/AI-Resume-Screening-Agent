"""Streamlit entry point for the AI Resume Screening Agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.candidate_result import CandidateResult
from src.screening_pipeline import ScreeningFailure, ScreeningPipeline
from src.similarity_engine import ModelLoadError, SemanticSimilarityEngine
from src.scoring_engine import ScoringWeights
from src.utils import export_candidates_to_csv, export_candidates_to_json


st.set_page_config(page_title="AI Resume Screening Agent", page_icon="🧠", layout="wide")


def _get_cached_semantic_engine() -> SemanticSimilarityEngine:
    @st.cache_resource
    def _load_engine() -> SemanticSimilarityEngine:
        return SemanticSimilarityEngine()

    return _load_engine()


def calculate_summary_metrics(candidates: list[CandidateResult], threshold: float) -> dict[str, float | int]:
    """Compute summary metrics for display in the UI."""
    if not candidates:
        return {"success_count": 0, "failed_count": 0, "average_score": 0.0, "strong_matches": 0, "threshold_matches": 0}

    average_score = sum(candidate.final_score for candidate in candidates) / len(candidates)
    strong_matches = sum(1 for candidate in candidates if candidate.final_score >= 80)
    threshold_matches = sum(1 for candidate in candidates if candidate.final_score >= threshold)
    return {
        "success_count": len(candidates),
        "failed_count": 0,
        "average_score": round(average_score, 2),
        "strong_matches": strong_matches,
        "threshold_matches": threshold_matches,
    }


def build_results_table(candidates: list[CandidateResult]) -> pd.DataFrame:
    """Create a display-friendly DataFrame for the ranked results table."""
    rows = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_rank = candidate.rank or index
        rows.append(
            {
                "Rank": candidate_rank,
                "Candidate": candidate.candidate_name,
                "Overall Score": round(candidate.final_score, 2),
                "Semantic Score": round(candidate.semantic_relevance_score, 2),
                "Skill Match": round(candidate.skill_match_score, 2),
                "Experience Match": round(candidate.experience_match_score, 2),
                "Education Match": round(candidate.education_match_score, 2),
                "Recommendation": candidate.recommendation,
                "Filename": candidate.original_filename,
            }
        )
    return pd.DataFrame(rows)


def _render_weight_config() -> tuple[ScoringWeights, bool]:
    st.sidebar.header("Application information")
    st.sidebar.write("- Resume parsing for PDF, DOCX, and TXT files")
    st.sidebar.write("- NLP semantic relevance using sentence-transformers")
    st.sidebar.write("- Skill, experience, and education matching")
    st.sidebar.write("- Explainable candidate ranking and decision support")

    st.sidebar.header("Scoring weights")
    semantic_weight = st.sidebar.slider("Semantic relevance (%)", 0, 100, 50)
    skill_weight = st.sidebar.slider("Required skill match (%)", 0, 100, 30)
    experience_weight = st.sidebar.slider("Experience relevance (%)", 0, 100, 15)
    education_weight = st.sidebar.slider("Education relevance (%)", 0, 100, 5)
    total_weight = semantic_weight + skill_weight + experience_weight + education_weight

    st.sidebar.write(f"Current total: {total_weight}%")
    if total_weight == 100:
        st.sidebar.success("Weights total 100% and are ready to use.")
    else:
        st.sidebar.warning("Weights must total 100% before screening can start.")

    try:
        weights = ScoringWeights(
            semantic=float(semantic_weight),
            skill=float(skill_weight),
            experience=float(experience_weight),
            education=float(education_weight),
        )
    except ValueError as exc:
        st.sidebar.error(str(exc))
        return ScoringWeights(), False

    return weights, total_weight == 100


def _render_threshold_config() -> float:
    st.sidebar.header("Shortlist threshold")
    threshold = st.sidebar.slider("Threshold (%)", 0, 100, 80)
    st.sidebar.caption("This is a decision-support setting and not an automatic hiring rule.")
    return float(threshold)


def _read_job_description() -> tuple[str, bool]:
    st.header("Step 1 — Add Job Description")
    pasted_text = st.text_area("Paste a job description", height=220)
    uploaded_jd = st.file_uploader("Or upload a TXT job description", type=["txt"], accept_multiple_files=False)

    if pasted_text and uploaded_jd is not None:
        st.info("Pasted text takes priority because it was provided.")
    if uploaded_jd is not None:
        try:
            content = uploaded_jd.read().decode("utf-8")
        except UnicodeDecodeError:
            st.error("The uploaded job description could not be decoded as UTF-8.")
            return "", False
        if content.strip():
            pasted_text = content

    if pasted_text is None:
        pasted_text = ""
    cleaned = pasted_text.strip()
    if not cleaned:
        st.warning("Enter a job description before running screening.")
        return "", False

    st.caption(f"Character count: {len(cleaned)}")
    with st.expander("Preview job description", expanded=False):
        st.write(cleaned)
    return cleaned, True


def _read_uploaded_resumes() -> list[Path]:
    st.header("Step 2 — Upload Candidate Resumes")
    uploaded_files = st.file_uploader("Upload one or more resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if not uploaded_files:
        st.info("No resumes selected yet.")
        return []

    st.caption(f"Selected files: {len(uploaded_files)}")
    st.caption("Supported formats: PDF, DOCX, TXT")
    st.caption("Scanned or image-only PDFs are not supported in this version because OCR is outside the current scope.")
    for uploaded_file in uploaded_files:
        st.write(f"- {uploaded_file.name}")

    temp_dir = Path(".tmp_uploads")
    temp_dir.mkdir(exist_ok=True)
    saved_paths: list[Path] = []
    for uploaded_file in uploaded_files:
        target_path = temp_dir / uploaded_file.name
        if target_path.exists():
            target_path = temp_dir / f"{uploaded_file.name}_{len(saved_paths)}"
        target_path.write_bytes(uploaded_file.getvalue())
        saved_paths.append(target_path)
    return saved_paths


def _run_screening(job_description: str, resume_paths: list[Path], weights: ScoringWeights) -> tuple[list[CandidateResult], list[ScreeningFailure], dict[str, Any]]:
    engine = _get_cached_semantic_engine()
    pipeline = ScreeningPipeline(job_description=job_description, semantic_engine=engine)
    pipeline.weights = weights
    results, failures = pipeline.run(resume_paths)
    summary = calculate_summary_metrics(results, threshold=80)
    summary["failed_count"] = len(failures)
    return results, failures, summary


def _render_results(results: list[CandidateResult], failures: list[ScreeningFailure], threshold: float) -> None:
    st.header("Screening Results")
    if not results and not failures:
        st.warning("No screening results were produced.")
        return

    summary = calculate_summary_metrics(results, threshold=threshold)
    summary["failed_count"] = len(failures)
    cols = st.columns(4)
    cols[0].metric("Successfully processed resumes", summary["success_count"])
    cols[1].metric("Failed files", summary["failed_count"])
    cols[2].metric("Average candidate score", f"{summary['average_score']:.2f}")
    cols[3].metric("Strong matches", summary["strong_matches"])

    if failures:
        st.warning("Some files could not be processed. Review the failure list below.")
        for failure in failures:
            st.write(f"- {failure.filename}: {failure.reason}")

    if not results:
        st.error("All uploaded files failed processing. Review the failures above and try again with valid resumes.")
        return

    table = build_results_table(results)
    st.dataframe(table, use_container_width=True)

    try:
        import plotly.express as px
    except Exception:
        px = None

    if px is not None:
        chart_frame = table[["Candidate", "Overall Score", "Filename"]].copy()
        chart_frame["Display Label"] = chart_frame["Candidate"]
        chart_frame["Display Label"] = chart_frame.apply(
            lambda row: f"{row['Candidate']} ({row['Filename']})" if row["Candidate"] in chart_frame["Candidate"].value_counts().index[chart_frame["Candidate"].value_counts() > 1] else row["Candidate"],
            axis=1,
        )
        fig = px.bar(
            chart_frame,
            x="Overall Score",
            y="Display Label",
            orientation="h",
            labels={"Overall Score": "Overall Relevance Score", "Display Label": "Candidate"},
            color="Overall Score",
        )
        fig.update_layout(yaxis={'autorange': 'reversed'})
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Candidate details", expanded=True):
        for candidate in results:
            st.subheader(f"{candidate.rank}. {candidate.candidate_name}")
            st.write(f"Original filename: {candidate.original_filename}")
            if candidate.email:
                st.write(f"Email: {candidate.email}")
            st.write(f"Overall score: {candidate.final_score:.2f}")
            st.write(f"Recommendation: {candidate.recommendation}")
            st.write(f"Semantic relevance score: {candidate.semantic_relevance_score:.2f}")
            st.write(f"Skill-match score: {candidate.skill_match_score:.2f}")
            st.write(f"Experience-match score: {candidate.experience_match_score:.2f}")
            st.write(f"Education-match score: {candidate.education_match_score:.2f}")
            st.write(f"Extracted experience: {candidate.extracted_experience if candidate.extracted_experience is not None else 'Not identified'}")
            st.write(f"Required experience: {candidate.required_experience if candidate.required_experience is not None else 'Not specified in the job description'}")
            st.write(f"Extracted education: {', '.join(candidate.extracted_education) if candidate.extracted_education else 'Not identified'}")
            st.write(f"Required education: {', '.join(candidate.required_education) if candidate.required_education else 'Not specified in the job description'}")
            st.write(f"Extracted technical skills: {', '.join(candidate.extracted_skills) if candidate.extracted_skills else 'Not identified'}")
            st.write(f"Matched required skills: {', '.join(candidate.matched_skills) if candidate.matched_skills else 'Not identified'}")
            st.write(f"Missing required skills: {', '.join(candidate.missing_skills) if candidate.missing_skills else 'Not identified'}")
            st.write(f"Strengths: {', '.join(candidate.strengths) if candidate.strengths else 'Not identified'}")
            st.write(f"Gaps: {', '.join(candidate.gaps) if candidate.gaps else 'Not identified'}")
            st.write(f"Explanation: {candidate.explanation}")
            st.write(f"Scoring notes: {', '.join(candidate.scoring_notes) if candidate.scoring_notes else 'Not identified'}")

    st.download_button(
        label="Download Ranked Results as CSV",
        data=export_candidates_to_csv(results),
        file_name="ranked_candidates.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download Ranked Results as JSON",
        data=export_candidates_to_json(results),
        file_name="ranked_candidates.json",
        mime="application/json",
    )

    with st.expander("Responsible AI and Limitations", expanded=False):
        st.write("This is a decision-support system. It does not replace recruiter judgment.")
        st.write("Semantic relevance is not candidate potential.")
        st.write("Resume wording can influence scores.")
        st.write("Rule-based extraction may make mistakes.")
        st.write("Scanned PDFs are not currently supported.")
        st.write("Protected and sensitive characteristics are excluded from scoring.")
        st.write("Recommendation thresholds are configurable defaults, not universal hiring rules.")
        st.write("Human review is required before hiring decisions.")


def main() -> None:
    st.title("AI Resume Screening Agent")
    st.caption("Explainable NLP-based candidate ranking and hiring decision support")
    st.info("This system is a decision-support tool. It should not automatically reject candidates or replace human hiring judgment.")
    st.write("Scores measure relevance to the supplied job description. They are not hiring probabilities, and human review is required.")

    weights, valid_weights = _render_weight_config()
    threshold = _render_threshold_config()

    job_description, jd_ready = _read_job_description()
    resume_paths = _read_uploaded_resumes()

    if st.button("Screen Candidates", type="primary"):
        if not jd_ready:
            st.error("Please provide a non-empty job description.")
        elif not resume_paths:
            st.error("Please upload at least one resume.")
        elif not valid_weights:
            st.error("Scoring weights must total 100% before screening can start.")
        else:
            with st.spinner("Screening candidates..."):
                try:
                    results, failures, _ = _run_screening(job_description, resume_paths, weights)
                except ModelLoadError as exc:
                    st.error(f"The semantic model could not be loaded: {exc}")
                    return
                except Exception as exc:  # pragma: no cover - defensive UI branch
                    st.error("The screening workflow could not be completed. Please review the uploaded files and try again.")
                    return
            st.session_state["results"] = results
            st.session_state["failures"] = failures
            st.session_state["threshold"] = threshold
            _render_results(results, failures, threshold)
            return

    if "results" in st.session_state:
        _render_results(st.session_state.get("results", []), st.session_state.get("failures", []), st.session_state.get("threshold", threshold))


if __name__ == "__main__":
    main()
