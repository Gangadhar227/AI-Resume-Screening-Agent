from __future__ import annotations

from pathlib import Path

import pytest

from src.resume_parser import ResumeParsingError, clean_extracted_text, extract_text_from_file


def test_extract_text_from_txt(tmp_path: Path) -> None:
    file_path = tmp_path / "candidate.txt"
    file_path.write_text("Alice Johnson\nPython developer\n", encoding="utf-8")

    text = extract_text_from_file(file_path)

    assert "Alice Johnson" in text
    assert "Python developer" in text


def test_extract_text_from_txt_utf8(tmp_path: Path) -> None:
    file_path = tmp_path / "candidate.txt"
    file_path.write_text("Café résumé — NLP engineer\n", encoding="utf-8")

    text = extract_text_from_file(file_path)

    assert "Café" in text
    assert "NLP" in text


def test_clean_extracted_text_removes_repeated_whitespace_and_blank_lines() -> None:
    raw_text = "Alice   Johnson\n\n\n\nPython   developer\n\n\nSkills:   NLP, ML"

    cleaned = clean_extracted_text(raw_text)

    assert "Alice Johnson" in cleaned
    assert "Python developer" in cleaned
    assert "Skills: NLP, ML" in cleaned
    assert "\n\n\n" not in cleaned


def test_empty_text_file_raises_helpful_error(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(ResumeParsingError, match="empty"):
        extract_text_from_file(file_path)


def test_unsupported_file_type_raises_helpful_error(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.md"
    file_path.write_text("hello", encoding="utf-8")

    with pytest.raises(ResumeParsingError, match="unsupported"):
        extract_text_from_file(file_path)


def test_invalid_pdf_raises_helpful_error(tmp_path: Path) -> None:
    file_path = tmp_path / "bad.pdf"
    file_path.write_bytes(b"not a real pdf")

    with pytest.raises(ResumeParsingError):
        extract_text_from_file(file_path)
