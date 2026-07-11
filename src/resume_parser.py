"""Resume parsing helpers for PDF, DOCX, and TXT files."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import fitz  # type: ignore
from docx import Document  # type: ignore


class ResumeParsingError(ValueError):
    """Raised when a resume file cannot be parsed into readable text."""


def clean_extracted_text(text: str) -> str:
    """Normalize whitespace while preserving useful line breaks."""
    if text is None:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n +", "\n", text)
    text = re.sub(r" +\n", "\n", text)
    text = text.strip()
    return text


def extract_text_from_pdf(file_path: str | Path) -> str:
    """Extract text from a PDF file. Returns a helpful error for scanned or unreadable PDFs."""
    path = Path(file_path)
    if not path.exists():
        raise ResumeParsingError("The file could not be found.")

    try:
        document = fitz.open(path)
    except Exception as exc:  # pragma: no cover - defensive branch
        raise ResumeParsingError("The PDF file appears to be corrupted or unreadable.") from exc

    if document.is_encrypted:
        raise ResumeParsingError("The PDF is password-protected and cannot be read.")

    text_parts: list[str] = []
    for page in document:
        page_text = page.get_text("text")
        if page_text:
            text_parts.append(page_text)

    document.close()

    if not text_parts:
        raise ResumeParsingError(
            "The PDF contains no extractable text. Scanned PDFs require OCR and are not supported in this version."
        )

    return clean_extracted_text("\n\n".join(text_parts))


def extract_text_from_docx(file_path: str | Path) -> str:
    """Extract text from a DOCX file."""
    path = Path(file_path)
    if not path.exists():
        raise ResumeParsingError("The file could not be found.")

    try:
        document = Document(path)
    except Exception as exc:  # pragma: no cover - defensive branch
        raise ResumeParsingError("The DOCX file appears to be corrupted or unreadable.") from exc

    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text and paragraph.text.strip()]
    text = "\n".join(paragraphs)
    if not text.strip():
        raise ResumeParsingError("The DOCX file is empty or contains no readable text.")
    return clean_extracted_text(text)


def extract_text_from_txt(file_path: str | Path) -> str:
    """Extract text from a plain text file using safe decoding."""
    path = Path(file_path)
    if not path.exists():
        raise ResumeParsingError("The file could not be found.")

    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise ResumeParsingError("The text file could not be read.") from exc

    if not raw_bytes.strip():
        raise ResumeParsingError("The text file is empty.")

    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            text = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ResumeParsingError("The text file could not be decoded as UTF-8 or UTF-16.")

    cleaned = clean_extracted_text(text)
    if not cleaned:
        raise ResumeParsingError("The text file is empty after cleaning.")
    return cleaned


def extract_text_from_file(file_path: str | Path) -> str:
    """Extract text from a supported file type, returning a user-friendly error when parsing fails."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix == ".docx":
        return extract_text_from_docx(path)
    if suffix in {".txt", ".text"}:
        return extract_text_from_txt(path)

    raise ResumeParsingError("unsupported file type. Please upload a PDF, DOCX, or TXT file.")
