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

EDUCATION_ALIASES: dict[str, str] = {
    "b.tech": "B.Tech",
    "b.e": "B.Tech",
    "b.e.": "B.Tech",
    "bachelor of technology": "B.Tech",
    "bachelor of engineering": "B.Tech",
    "bachelor's degree": "Bachelor",
    "bachelor’s degree": "Bachelor",
    "bachelor degree": "Bachelor",
    "bachelor": "Bachelor",
    "bs": "Bachelor",
    "b.s.": "Bachelor",
    "b.s": "Bachelor",
    "m.tech": "M.Tech",
    "master of technology": "M.Tech",
    "master's degree": "Master",
    "master’s degree": "Master",
    "master degree": "Master",
    "master": "Master",
    "ms": "Master",
    "m.s.": "Master",
    "m.s": "Master",
    "mca": "MCA",
    "bca": "BCA",
    "phd": "PhD",
    "m.sc": "M.Sc",
    "b.sc": "B.Sc",
    "mba": "MBA",
}

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
    r"\bBS\b",
    r"\bB\.S\.\b",
    r"\bB\.S\b",
    r"\bMS\b",
    r"\bM\.S\.\b",
    r"\bM\.S\b",
    r"\bBachelor(?:'s|’s)?\s+degree\b",
    r"\bBachelor\s+degree\b",
    r"\bBachelor\b",
    r"\bMaster(?:'s|’s)?\s+degree\b",
    r"\bMaster\s+degree\b",
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


def canonicalize_skill(skill: str) -> str:
    if not skill:
        return ""
    cleaned = skill.strip()
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    if lowered in SKILL_ALIASES:
        return SKILL_ALIASES[lowered]
    for alias, canonical in SKILL_ALIASES.items():
        if alias.lower() == lowered:
            return canonical
    for canonical in SKILL_ORDER:
        if canonical.lower() == lowered:
            return canonical
    return cleaned


def normalize_skills(skills: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for skill in skills:
        canonical = canonicalize_skill(skill)
        if canonical and canonical not in seen:
            seen.add(canonical)
            ordered.append(canonical)
    return ordered


def canonicalize_education(qualification: str, prefer_degree_labels: bool = False) -> str:
    if not qualification:
        return ""
    cleaned = qualification.strip()
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    
    canonical = cleaned
    if lowered in EDUCATION_ALIASES:
        canonical = EDUCATION_ALIASES[lowered]
    else:
        for alias, can in EDUCATION_ALIASES.items():
            if alias.lower() == lowered:
                canonical = can
                break
                
    if prefer_degree_labels:
        if canonical in {"B.Tech", "B.E.", "B.Sc", "BCA", "BS", "Bachelor"}:
            return "Bachelor"
        if canonical in {"M.Tech", "M.Sc", "MBA", "MCA", "MS", "Master"}:
            return "Master"
            
    return canonical


def normalize_education(qualifications: list[str], prefer_degree_labels: bool = False) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    
    canonical_list = []
    for qualification in qualifications:
        canonical = canonicalize_education(qualification, prefer_degree_labels=prefer_degree_labels)
        if canonical and canonical not in canonical_list:
            canonical_list.append(canonical)
            
    has_bachelor = "Bachelor" in canonical_list
    has_master = "Master" in canonical_list
    
    bachelor_specific = {"B.Tech", "B.E.", "B.Sc", "BCA", "BS"}
    master_specific = {"M.Tech", "M.Sc", "MBA", "MCA", "MS"}
    
    for canonical in canonical_list:
        if has_bachelor and canonical in bachelor_specific:
            continue
        if has_master and canonical in master_specific:
            continue
        if canonical not in seen:
            seen.add(canonical)
            ordered.append(canonical)
            
    return ordered


def _extract_skills(text: str) -> list[str]:
    normalized_text = re.sub(r"[^A-Za-z0-9+.#()/-]+", " ", text)
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
    return normalize_skills(ordered)


def _extract_education(text: str) -> list[str]:
    education_matches: list[str] = []
    seen: set[str] = set()
    prefer_degree_labels = bool(
        re.search(
            r"\bBachelor(?:'s|’s)?\s+degree\b|\bBachelor\s+degree\b|\bBachelor\b|\bMaster(?:'s|’s)?\s+degree\b|\bMaster\s+degree\b|\bMaster\b",
            text,
            flags=re.IGNORECASE,
        )
    )
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
            canonical = canonicalize_education(qualification, prefer_degree_labels=prefer_degree_labels)
            if canonical and canonical not in seen:
                education_matches.append(canonical)
                seen.add(canonical)
    return normalize_education(education_matches, prefer_degree_labels=prefer_degree_labels)


def _extract_experience(text: str) -> tuple[float | None, str | None]:
    # 1. Check for compound year and month pattern, e.g. "1 year and 6 months" or "2 years 3 months"
    compound_pattern = r"(?<!\d)(\d+(?:\.\d+)?)\s*years?\s*(?:and\s*)?(\d+(?:\.\d+)?)\s*months?"
    match = re.search(compound_pattern, text, flags=re.IGNORECASE)
    if match:
        years = float(match.group(1))
        months = float(match.group(2))
        return round(years + (months / 12.0), 2), None

    # 2. Check for years only, e.g. "2.5 years of experience", "3 years", etc.
    years_pattern = r"(?<!\d)(\d+(?:\.\d+)?)\s*years?"
    match = re.search(years_pattern, text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1)), None

    # 3. Check for months only (representing internships/short-term experience), e.g. "6-month internship", "6 months", "3-month"
    months_pattern = r"(?<!\d)(\d+(?:\.\d+)?)\s*[-–—]?\s*months?"
    match = re.search(months_pattern, text, flags=re.IGNORECASE)
    if match:
        months = float(match.group(1))
        return round(months / 12.0, 2), "Approximate internship duration used as relevant experience."

    return None, None


def extract_candidate_information(text: str, filename: str | None = None) -> dict[str, Any]:
    """Extract structured candidate information from resume text using transparent heuristics."""
    if text is None:
        text = ""

    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    collapsed_text = re.sub(r"\s+", " ", normalized_text).strip()
    experience_years, experience_note = _extract_experience(collapsed_text)
    return {
        "name": _extract_name(normalized_text, filename=filename),
        "email": _extract_email(collapsed_text),
        "skills": _extract_skills(collapsed_text),
        "education": _extract_education(collapsed_text),
        "experience_years": experience_years,
        "experience_note": experience_note,
    }
