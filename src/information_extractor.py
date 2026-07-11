"""Rule-based extraction of candidate metadata from resume text."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


SKILL_ALIASES: dict[str, str] = {
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "nlp": "Natural Language Processing",
    "natural language processing": "Natural Language Processing",
    "nlu": "Natural Language Understanding",
    "computer vision": "Computer Vision",
    "cv": "Computer Vision",
    "data science": "Data Science",
    "generative ai": "Generative AI",
    "genai": "Generative AI",
    "large language models": "Large Language Models",
    "large language model": "Large Language Models",
    "llm": "Large Language Models",
    "llms": "Large Language Models",
    "rag": "Retrieval-Augmented Generation",
    "retrieval augmented generation": "Retrieval-Augmented Generation",
    "prompt engineering": "Prompt Engineering",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "scikit-learn": "Scikit-learn",
    "sklearn": "Scikit-learn",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "opencv": "OpenCV",
    "yolo": "YOLO",
    "langchain": "LangChain",
    "flask": "Flask",
    "django": "Django",
    "fastapi": "FastAPI",
    "streamlit": "Streamlit",
    "sql": "SQL",
    "mysql": "MySQL",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "mongodb": "MongoDB",
    "sqlite": "SQLite",
    "vector database": "Vector Database",
    "faiss": "FAISS",
    "chromadb": "ChromaDB",
    "aws": "AWS",
    "azure": "Azure",
    "gcp": "GCP",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "git": "Git",
    "github": "GitHub",
    "rest api": "REST API",
    "restful api": "REST API",
    "restful apis": "REST API",
    "rest apis": "REST API",
    "html": "HTML",
    "css": "CSS",
    "react": "React",
    "node.js": "Node.js",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "python": "Python",
    "java": "Java",
    "c++": "C++",
    "c": "C",
}

SKILL_ORDER: list[str] = [
    "Python",
    "Java",
    "C",
    "C++",
    "JavaScript",
    "Machine Learning",
    "Deep Learning",
    "Natural Language Processing",
    "Computer Vision",
    "Data Science",
    "Generative AI",
    "Large Language Models",
    "Retrieval-Augmented Generation",
    "Prompt Engineering",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "TensorFlow",
    "PyTorch",
    "OpenCV",
    "YOLO",
    "LangChain",
    "Flask",
    "Django",
    "FastAPI",
    "Streamlit",
    "SQL",
    "MySQL",
    "PostgreSQL",
    "MongoDB",
    "SQLite",
    "Vector Database",
    "FAISS",
    "ChromaDB",
    "AWS",
    "Azure",
    "GCP",
    "Docker",
    "Kubernetes",
    "Git",
    "GitHub",
    "REST API",
    "HTML",
    "CSS",
    "React",
    "Node.js",
]

EDUCATION_PATTERNS = [
    r"\bB\.Tech\b",
    r"\bBachelor of Technology\b",
    r"\bB\.E\.?\b",
    r"\bBachelor of Engineering\b",
    r"\bB\.Sc\b",
    r"\bBCA\b",
    r"\bMCA\b",
    r"\bM\.Tech\b",
    r"\bMaster of Technology\b",
    r"\bM\.Sc\b",
    r"\bMBA\b",
    r"\bPhD\b",
    r"\bBachelor\b",
    r"\bMaster\b",
]

NAME_BLOCKLIST = {
    "resume",
    "curriculum vitae",
    "cv",
    "profile",
    "summary",
    "contact",
}


def _normalize_name(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _extract_name(text: str, filename: str | None = None) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:8]:
        candidate = re.sub(r"\s+", " ", line)
        candidate_lower = candidate.lower()
        if candidate_lower in NAME_BLOCKLIST:
            continue
        if re.search(r"@|http|\d{3}-?\d{3}|tel:|phone", candidate_lower):
            continue
        if re.fullmatch(r"[A-Za-z .,'-]+", candidate) is None:
            continue
        if len(candidate.split()) < 2 or len(candidate.split()) > 4:
            continue
        if candidate_lower.startswith(("education", "skills", "experience", "projects", "contact")):
            continue
        if candidate_lower in {"software engineer", "developer", "engineer", "analyst", "manager", "consultant"}:
            continue
        return _normalize_name(candidate)

    if filename:
        return Path(filename).stem
    return "Unknown"


def _extract_email(text: str) -> str | None:
    match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
    return match.group(0) if match else None


def _extract_skills(text: str) -> list[str]:
    normalized_text = re.sub(r"[^A-Za-z0-9+.#()/-]+", " ", text)
    words = normalized_text.split()
    found: set[str] = set()

    for item in SKILL_ORDER:
        if item in {"C", "C++"}:
            pattern = r"(?<![A-Za-z0-9])" + re.escape(item) + r"(?![A-Za-z0-9])"
        else:
            pattern = r"(?<![A-Za-z0-9])" + re.escape(item.lower()) + r"(?![A-Za-z0-9])"
        if re.search(pattern, normalized_text, flags=re.IGNORECASE):
            found.add(item)

    for alias, canonical in SKILL_ALIASES.items():
        if alias in {"c", "c++"}:
            pattern = r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])"
        else:
            pattern = r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])"
        if re.search(pattern, normalized_text, flags=re.IGNORECASE):
            found.add(canonical)

    ordered: list[str] = []
    for canonical in SKILL_ORDER:
        if canonical in found:
            ordered.append(canonical)
    for item in sorted(found):
        if item not in ordered:
            ordered.append(item)
    return ordered


def _extract_education(text: str) -> list[str]:
    education_matches: list[str] = []
    seen: set[str] = set()
    for pattern in EDUCATION_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            qualification = match.group(0)
            if qualification.lower() == "b.e":
                qualification = "B.E."
            elif qualification.lower() == "b.tech":
                qualification = "B.Tech"
            elif qualification.lower() == "m.tech":
                qualification = "M.Tech"
            elif qualification.lower() == "m.sc":
                qualification = "M.Sc"
            elif qualification.lower() == "b.sc":
                qualification = "B.Sc"
            elif qualification.lower() == "bachelor":
                qualification = "Bachelor"
            elif qualification.lower() == "master":
                qualification = "Master"
            if qualification not in seen:
                education_matches.append(qualification)
                seen.add(qualification)
    return education_matches


def _extract_experience(text: str) -> float | None:
    patterns = [
        r"(?:over|about|around)?\s*(\d+(?:\.\d+)?)\s*\+?\s*years?\s*(?:of\s+)?experience",
        r"(?:over|about|around)?\s*(\d+(?:\.\d+)?)\s*\+?\s*years?",
    ]
    matches: list[float] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = float(match.group(1))
            matches.append(value)
    if not matches:
        return None
    # Conservative rule: use the first explicit experience statement when multiple appear.
    return matches[0]


def extract_candidate_information(text: str, filename: str | None = None) -> dict[str, Any]:
    """Extract structured candidate information from resume text using transparent heuristics."""
    if text is None:
        text = ""

    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    collapsed_text = re.sub(r"\s+", " ", normalized_text).strip()
    return {
        "name": _extract_name(normalized_text, filename=filename),
        "email": _extract_email(collapsed_text),
        "skills": _extract_skills(collapsed_text),
        "education": _extract_education(collapsed_text),
        "experience_years": _extract_experience(collapsed_text),
    }
