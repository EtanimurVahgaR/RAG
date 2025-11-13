from __future__ import annotations
from io import BytesIO
from pathlib import Path
import tempfile
import os
from typing import Optional


def _safe_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


_pdfplumber = _safe_import("pdfplumber")
_fitz = _safe_import("fitz")  # PyMuPDF
_PyPDF2 = _safe_import("PyPDF2")
_docx = _safe_import("docx")


def read_pdf(file_bytes: bytes) -> str:
    # Prefer pdfplumber -> PyMuPDF -> PyPDF2
    if _pdfplumber is not None:
        try:
            texts = []
            with _pdfplumber.open(BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    texts.append(text)
            return "\n\n".join(texts).strip()
        except Exception:
            pass
    if _fitz is not None:
        try:
            doc = _fitz.open(stream=file_bytes, filetype="pdf")
            texts = [page.get_text("text") or "" for page in doc]
            return "\n\n".join(texts).strip()
        except Exception:
            pass
    if _PyPDF2 is not None:
        try:
            PdfReader = _PyPDF2.PdfReader
            reader = PdfReader(BytesIO(file_bytes))
            texts = []
            for page in reader.pages:
                try:
                    texts.append(page.extract_text() or "")
                except Exception:
                    texts.append("")
            return "\n\n".join(texts).strip()
        except Exception:
            pass
    return ""


def read_docx(file_bytes: bytes) -> str:
    if _docx is None:
        return ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        document = _docx.Document(tmp_path)
        paragraphs = [p.text for p in document.paragraphs]
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return "\n\n".join(paragraphs).strip()
    except Exception:
        return ""


def read_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="replace")
    except Exception:
        return ""


def extract_text(file_bytes: bytes, ext: str) -> str:
    ext = (ext or "").lower()
    if ext == ".pdf":
        return read_pdf(file_bytes)
    if ext == ".docx":
        return read_docx(file_bytes)
    if ext == ".txt":
        return read_txt(file_bytes)
    return ""
