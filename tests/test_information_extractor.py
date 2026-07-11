from __future__ import annotations

from src.information_extractor import extract_candidate_information


def test_email_extraction() -> None:
    text = "Contact: alice@example.com"
    info = extract_candidate_information(text, filename="alice")
    assert info["email"] == "alice@example.com"


def test_missing_email_returns_none() -> None:
    info = extract_candidate_information("No contact details here", filename="candidate")
    assert info["email"] is None


def test_candidate_name_extraction() -> None:
    text = "Alice Johnson\nSoftware Engineer"
    info = extract_candidate_information(text, filename="candidate")
    assert info["name"] == "Alice Johnson"


def test_candidate_name_fallback_uses_filename() -> None:
    info = extract_candidate_information("Summary\nExperienced developer", filename="candidate_01")
    assert info["name"] == "candidate_01"


def test_skill_extraction_and_aliases() -> None:
    text = "Experienced in Python, NLP, ML, sklearn, RAG, RESTful APIs, Postgres, and K8s."
    info = extract_candidate_information(text, filename="candidate")
    assert "Python" in info["skills"]
    assert "Natural Language Processing" in info["skills"]
    assert "Machine Learning" in info["skills"]
    assert "Scikit-learn" in info["skills"]
    assert "Retrieval-Augmented Generation" in info["skills"]
    assert "REST API" in info["skills"]
    assert "PostgreSQL" in info["skills"]
    assert "Kubernetes" in info["skills"]


def test_duplicate_skills_are_removed_and_order_is_deterministic() -> None:
    text = "Python, python, NLP, NLP"
    info = extract_candidate_information(text, filename="candidate")
    assert info["skills"] == ["Python", "Natural Language Processing"]


def test_case_insensitive_matching_and_substring_avoidance() -> None:
    text = "C candidate, C++ application, digital, candidate"
    info = extract_candidate_information(text, filename="candidate")
    assert "C" in info["skills"]
    assert "C++" in info["skills"]
    assert "Git" not in info["skills"]


def test_education_extraction() -> None:
    text = "Education: B.Tech in Computer Science, M.Sc in AI"
    info = extract_candidate_information(text, filename="candidate")
    assert "B.Tech" in info["education"]
    assert "M.Sc" in info["education"]


def test_experience_extraction_decimal() -> None:
    text = "Over 2.5 years of experience in NLP and Python."
    info = extract_candidate_information(text, filename="candidate")
    assert info["experience_years"] == 2.5


def test_experience_extraction_from_internship_duration() -> None:
    text = "Completed a 6-month internship in NLP and Python."
    info = extract_candidate_information(text, filename="candidate")
    assert info["experience_years"] == 0.5
    assert info["experience_note"] == "Approximate internship duration used as relevant experience."


def test_education_extraction_normalizes_equivalent_qualifications() -> None:
    text = "Bachelor's degree, B.Tech, Master of Technology, and M.Tech"
    info = extract_candidate_information(text, filename="candidate")
    assert info["education"] == ["Bachelor", "Master"]


def test_missing_experience_returns_none() -> None:
    info = extract_candidate_information("No experience statement here", filename="candidate")
    assert info["experience_years"] is None


def test_complete_structured_candidate_extraction() -> None:
    text = "Alice Johnson\nEmail: alice@example.com\nEducation: B.Tech in AI\nSkills: Python, NLP, SQL\nExperience: 3 years of experience"
    info = extract_candidate_information(text, filename="candidate")

    assert info["name"] == "Alice Johnson"
    assert info["email"] == "alice@example.com"
    assert info["education"] == ["B.Tech"]
    assert "Python" in info["skills"]
    assert "Natural Language Processing" in info["skills"]
    assert "SQL" in info["skills"]
    assert info["experience_years"] == 3.0
