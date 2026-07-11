from __future__ import annotations

from src.candidate_result import CandidateResult
from app import calculate_summary_metrics, build_results_table


def test_summary_metrics_are_calculated_from_results() -> None:
    candidates = [
        CandidateResult(candidate_name="A", original_filename="a.pdf", final_score=80.0, semantic_relevance_score=80.0, skill_match_score=80.0, experience_match_score=80.0, education_match_score=80.0, recommendation="Strong Match"),
        CandidateResult(candidate_name="B", original_filename="b.pdf", final_score=40.0, semantic_relevance_score=40.0, skill_match_score=40.0, experience_match_score=40.0, education_match_score=40.0, recommendation="Low Match"),
    ]
    metrics = calculate_summary_metrics(candidates, threshold=80)
    assert metrics["success_count"] == 2
    assert metrics["average_score"] == 60.0
    assert metrics["strong_matches"] == 1
    assert metrics["threshold_matches"] == 1


def test_results_table_contains_rank_and_candidate_columns() -> None:
    candidates = [CandidateResult(candidate_name="A", original_filename="a.pdf", final_score=80.0, semantic_relevance_score=80.0, skill_match_score=80.0, experience_match_score=80.0, education_match_score=80.0, recommendation="Strong Match")]
    table = build_results_table(candidates)
    assert table["Rank"][0] == 1
    assert table["Candidate"][0] == "A"
    assert table["Overall Score"][0] == 80.0
