"""
services/pdf_parser.py
======================
Turns an uploaded resume/cover-letter file (PDF or DOCX) into two things:

  1. the plain TEXT  -> fed to Engine A (the AI-content analyzer)
  2. the METADATA    -> the hidden "who/what created this file" fields

Why metadata matters
--------------------
Every PDF/DOCX stores the tool that produced it. A resume a human wrote in
Word says Producer="Microsoft Word". A resume dumped by an automation script
often says Producer="pdfkit", "wkhtmltopdf", "Puppeteer", or exports that
mention "ChatGPT" / "Skia" (headless Chrome). Those are strong, cheap signals
that something automated generated the document — so we surface them.

Everything here fails SAFELY: if a file is corrupt or a library is missing,
we return whatever we can plus a note, instead of crashing the whole request.
"""

from __future__ import annotations

import io
from typing import Tuple

# Tools that legitimate human-made documents almost never use. If any appears
# in the file's Producer/Creator/Author metadata, we raise a flag.
SUSPICIOUS_PRODUCERS = [
    "pdfkit",
    "wkhtmltopdf",
    "puppeteer",
    "headless",
    "chromium",
    "skia/pdf",  # headless Chrome's PDF engine
    "reportlab",  # common in scripted PDF generation
    "chatgpt",
    "openai",
    "weasyprint",
    "phantomjs",
    "selenium",
]


def _extract_pdf(data: bytes) -> Tuple[str, dict]:
    """Extract text + metadata from a PDF using pdfplumber."""
    import pdfplumber  # imported lazily so a missing lib doesn't break startup

    text_parts: list[str] = []
    metadata: dict = {}
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        # pdf.metadata is a dict like {"Producer": "...", "Creator": "...", ...}
        metadata = {k: str(v) for k, v in (pdf.metadata or {}).items()}
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts).strip(), metadata


def _extract_docx(data: bytes) -> Tuple[str, dict]:
    """Extract text + core metadata from a .docx using python-docx."""
    import docx  # python-docx

    document = docx.Document(io.BytesIO(data))
    text = "\n".join(p.text for p in document.paragraphs).strip()

    cp = document.core_properties
    metadata = {
        "Author": cp.author or "",
        "LastModifiedBy": cp.last_modified_by or "",
        "Created": str(cp.created) if cp.created else "",
        "Modified": str(cp.modified) if cp.modified else "",
        "Application": "",  # docx doesn't expose the app name via core props
    }
    return text, metadata


def extract_text_and_metadata(data: bytes, filename: str) -> Tuple[str, dict, list[str]]:
    """
    Main entry point.

    Returns a 3-tuple:
        text            (str)        -> the document's plain text
        metadata        (dict)       -> raw metadata fields
        metadata_flags  (list[str])  -> human-readable warnings, e.g.
                                        "PDF produced by 'pdfkit' (automation tool)"
    """
    name = (filename or "").lower()
    text, metadata = "", {}

    try:
        if name.endswith(".pdf"):
            text, metadata = _extract_pdf(data)
        elif name.endswith(".docx"):
            text, metadata = _extract_docx(data)
        elif name.endswith(".txt"):
            text = data.decode("utf-8", errors="ignore").strip()
        else:
            # Unknown extension: try UTF-8 as a last resort.
            text = data.decode("utf-8", errors="ignore").strip()
    except Exception as exc:  # never let a bad file kill the request
        return text, metadata, [f"Could not fully parse file: {exc}"]

    # ---- Turn raw metadata into human-readable FLAGS ----------------------
    flags: list[str] = []
    joined = " ".join(str(v).lower() for v in metadata.values())
    for tool in SUSPICIOUS_PRODUCERS:
        if tool in joined:
            flags.append(
                f"Document metadata references '{tool}', a tool commonly used "
                f"to auto-generate documents rather than author them by hand."
            )

    # A completely empty Author + Producer is mildly suspicious for a PDF.
    if name.endswith(".pdf") and not any(metadata.get(k) for k in ("Author", "Producer", "Creator")):
        flags.append("PDF has no Author/Producer/Creator metadata (often stripped by scripts).")

    return text, metadata, flags
