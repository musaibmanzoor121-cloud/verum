"""
main.py
=======
FastAPI entry point. It:
  * configures CORS (so the React frontend can call it from the browser),
  * installs the bot-detection middleware,
  * exposes the endpoints, and
  * assembles the three engines' outputs into ONE explainable report.

Run locally:
    uvicorn app.main:app --reload --port 8000
Interactive API docs are then at http://localhost:8000/docs
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.middleware.bot_detector import BotDetectionMiddleware
from app.schemas import (
    AuthenticityReport,
    BehaviorAnalysis,
    BehaviorPayload,
    EvidenceItem,
    TextAnalyzeRequest,
)
from app.services import ai_analyzer, behavior_scorer, consistency_checker, pdf_parser

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Dual-engine applicant authenticity screening with explainable evidence.",
)

# CORS: allow the frontend origin(s) to call this API from a browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Our custom middleware (rate limiting + fingerprint capture).
app.add_middleware(BotDetectionMiddleware)

# Order used to sort merged evidence so the scariest items float to the top.
_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}

LIMITATIONS_NOTE = (
    "Verum produces confidence bands, not verdicts. No statistical detector is "
    "foolproof: a careful adversarial user can reduce accuracy, and behavioral "
    "signals can misfire for users of assistive technology. Use these flags to "
    "prioritize HUMAN review — never to auto-reject a candidate."
)


def _assemble_report(
    resume_text: str,
    cover_letter_text: str,
    behavior: Optional[BehaviorPayload],
    metadata_flags: List[str],
    user_agent: str,
    client_ip: str,
    started: float,
) -> AuthenticityReport:
    """Run all three engines and fuse them into a single report."""
    # Engine A — content authenticity. We analyze the applicant's writing as a
    # WHOLE (resume + cover letter). A ChatGPT-written cover letter (flowing
    # prose) is far more detectable than a terse bullet-point resume, so folding
    # both in gives the content engine much more to work with.
    content_text = resume_text
    if cover_letter_text and cover_letter_text.strip():
        content_text = f"{resume_text}\n\n{cover_letter_text}"
    content = ai_analyzer.analyze_content(content_text, metadata_flags)
    # Cross-document consistency
    consistency = consistency_checker.compare_documents(resume_text, cover_letter_text)
    # Engine B — behavior / bot
    behavior_result = behavior_scorer.score_behavior(behavior, user_agent, client_ip)

    # Merge and rank all evidence.
    all_evidence: List[EvidenceItem] = (
        list(content.evidence) + list(consistency.evidence) + list(behavior_result.evidence)
    )
    all_evidence.sort(key=lambda e: _SEVERITY_RANK.get(e.severity, 3))

    # Fuse the sub-scores into one overall concern level.
    inconsistency = (1.0 - consistency.consistency_score) if consistency.checked else 0.0
    bot = behavior_result.bot_likelihood_score
    ai = content.ai_likelihood_score

    # A weighted blend is the baseline reading...
    blend = 0.45 * bot + 0.40 * ai + 0.15 * inconsistency

    # ...but a plain average has a dangerous failure mode: when a *human*
    # submits AI-written text, the bot score is legitimately ~0 and drags the
    # average below the review line — hiding obvious AI writing. These are
    # INDEPENDENT concerns: AI-written text is worth a look even if a real
    # person clicked submit. So the overall concern is never less than what the
    # single strongest engine says (scaled slightly, so one engine must be
    # fairly confident to trigger on its own). Whichever is higher wins.
    overall = max(
        blend,
        0.85 * ai,   # confident "this text is AI-written" alone -> human review
        0.90 * bot,  # confident "this was a bot/script" alone -> human review
    )
    overall = max(0.0, min(1.0, overall))

    verdict, recommendation = _verdict_for(overall)

    return AuthenticityReport(
        verdict=verdict,
        recommendation=recommendation,
        overall_flag_score=round(overall, 3),
        content=content,
        consistency=consistency,
        behavior=behavior_result,
        evidence=all_evidence,
        limitations=LIMITATIONS_NOTE,
        processing_ms=round((time.perf_counter() - started) * 1000, 1),
    )


def _verdict_for(overall: float) -> tuple[str, str]:
    """Map the overall score to a human verdict + a review recommendation."""
    if overall < 0.34:
        return (
            "Likely authentic",
            "No strong flags. Proceed with normal review.",
        )
    if overall < 0.66:
        return (
            "Needs human review",
            "Some authenticity signals were raised. Have a human review the "
            "flagged evidence below before advancing or rejecting.",
        )
    return (
        "Strong AI/automation signals",
        "Multiple strong flags. Prioritize for careful human review — do NOT "
        "auto-reject; confirm with the evidence and, if needed, a short live task.",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
# If a built frontend has been copied into ../static (the Docker build does
# this), we serve the React SPA from "/". Otherwise "/" returns JSON API info.
# Either way, the API always stays under /api and /health.
_STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


if not _STATIC_DIR.is_dir():

    @app.get("/")
    def root():
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "endpoints": ["/health", "/api/analyze-text", "/api/analyze"],
        }


@app.get("/health")
def health():
    return {"status": "ok", "transformer_perplexity": settings.USE_TRANSFORMER_PERPLEXITY}


@app.post("/api/analyze-text", response_model=AuthenticityReport)
def analyze_text(payload: TextAnalyzeRequest, request: Request):
    """JSON-only analysis (no file upload). Ideal for quick testing/demos."""
    started = time.perf_counter()
    return _assemble_report(
        resume_text=payload.resume_text,
        cover_letter_text=payload.cover_letter_text,
        behavior=payload.behavior,
        metadata_flags=[],  # no file => no document metadata
        user_agent=getattr(request.state, "user_agent", ""),
        client_ip=getattr(request.state, "client_ip", ""),
        started=started,
    )


@app.post("/api/analyze", response_model=AuthenticityReport)
async def analyze(
    request: Request,
    resume: UploadFile = File(..., description="Resume PDF/DOCX/TXT"),
    cover_letter: Optional[UploadFile] = File(None, description="Optional cover letter"),
    behavior: Optional[str] = Form(None, description="Behavior payload as a JSON string"),
):
    """Full analysis from uploaded files + optional behavioral biometrics."""
    started = time.perf_counter()

    resume_bytes = await resume.read()
    resume_text, _meta, resume_flags = pdf_parser.extract_text_and_metadata(
        resume_bytes, resume.filename
    )

    cover_text, cover_flags = "", []
    if cover_letter is not None:
        cover_bytes = await cover_letter.read()
        cover_text, _cmeta, cover_flags = pdf_parser.extract_text_and_metadata(
            cover_bytes, cover_letter.filename
        )

    # Parse the behavior JSON string (sent as a multipart form field).
    behavior_obj: Optional[BehaviorPayload] = None
    if behavior:
        try:
            behavior_obj = BehaviorPayload(**json.loads(behavior))
        except Exception:
            behavior_obj = None  # malformed behavior data is simply ignored

    return _assemble_report(
        resume_text=resume_text,
        cover_letter_text=cover_text,
        behavior=behavior_obj,
        metadata_flags=resume_flags + cover_flags,
        user_agent=getattr(request.state, "user_agent", ""),
        client_ip=getattr(request.state, "client_ip", ""),
        started=started,
    )


# Mount the built SPA LAST so all /api and /health routes above win. With
# html=True, StaticFiles serves index.html for unknown paths (SPA routing).
if _STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="spa")
