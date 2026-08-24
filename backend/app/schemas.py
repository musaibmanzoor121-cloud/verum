"""
schemas.py
==========
Pydantic models = the "shape" of the data going in and out of the API.

Why Pydantic?
-------------
FastAPI uses these classes to (1) validate incoming JSON automatically,
(2) generate interactive API docs at /docs, and (3) serialize responses.
If a client sends the wrong type, FastAPI rejects it with a clear 422 error
BEFORE your code ever runs. That's a huge safety net for free.

Two groups of models below:
  * REQUEST models  — what the client sends us (behavior data, text).
  * RESPONSE models — the authenticity report we send back.

Design principle from the project brief: never output a bare number. Every
signal carries EVIDENCE (an excerpt or a concrete explanation) and a
confidence BAND, not a fake-precise score.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# REQUEST MODELS  (client -> server)
# ---------------------------------------------------------------------------
class MouseEvent(BaseModel):
    """A single sampled mouse position. `t` is milliseconds since form load."""

    t: float
    x: float
    y: float


class BehaviorPayload(BaseModel):
    """
    Behavioral biometrics captured client-side while the applicant fills the
    form. We deliberately DO NOT capture which keys were pressed (privacy) —
    only the timing *between* keystrokes, which is enough to measure rhythm.
    """

    form_load_time: float = Field(..., description="epoch ms when the form loaded")
    submit_time: float = Field(..., description="epoch ms when submit was clicked")
    mouse_events: List[MouseEvent] = Field(default_factory=list)
    key_intervals_ms: List[float] = Field(
        default_factory=list, description="ms gaps between consecutive keystrokes"
    )
    scroll_events: List[float] = Field(
        default_factory=list, description="timestamps (ms) of scroll actions"
    )
    honeypot: str = Field(
        default="", description="hidden field; must stay empty for real humans"
    )


class TextAnalyzeRequest(BaseModel):
    """
    JSON-only convenience endpoint input (no file upload needed). Great for
    quick testing with curl and for the frontend's paste-text mode.
    """

    resume_text: str = Field(..., min_length=1)
    cover_letter_text: str = ""
    behavior: Optional[BehaviorPayload] = None


# ---------------------------------------------------------------------------
# RESPONSE MODELS  (server -> client)
# ---------------------------------------------------------------------------
class EvidenceItem(BaseModel):
    """
    The heart of the "explainability layer". Every flag the recruiter sees is
    one of these — it names the engine, the signal, a human explanation, a
    severity, and (when relevant) the exact text excerpt that triggered it.
    """

    engine: str  # "content" | "consistency" | "behavior" | "metadata"
    signal: str  # short machine-ish label, e.g. "low_burstiness"
    detail: str  # human-readable explanation for the recruiter
    severity: str  # "low" | "medium" | "high"
    excerpt: Optional[str] = None  # exact sentence/phrase, if applicable


class ContentAnalysis(BaseModel):
    """Engine A output: is the WRITING likely AI-generated?"""

    ai_likelihood_band: str  # e.g. "60–75% likely AI-assisted"
    ai_likelihood_score: float  # 0..1 internal score behind the band
    perplexity: Optional[float] = None
    perplexity_method: str  # "gpt2" or "heuristic"
    burstiness: float  # variance in sentence length (higher = more human)
    buzzword_density: float  # buzzword hits per 100 words
    buzzwords_found: List[str] = []
    metadata_flags: List[str] = []
    evidence: List[EvidenceItem] = []


class ConsistencyAnalysis(BaseModel):
    """NEW engine: do the resume and cover letter tell the SAME story?"""

    consistency_score: float  # 0..1, higher = more consistent
    checked: bool  # False if no cover letter was provided
    mismatches: List[str] = []
    evidence: List[EvidenceItem] = []


class BehaviorAnalysis(BaseModel):
    """Engine B output: was this submitted by a HUMAN or a SCRIPT?"""

    bot_likelihood_band: str
    bot_likelihood_score: float  # 0..1
    honeypot_triggered: bool
    time_to_submit_seconds: Optional[float] = None
    suspicious_user_agent: bool
    mouse_entropy: Optional[float] = None  # higher = more human-like
    keystroke_regularity: Optional[float] = None  # higher = more robotic
    evidence: List[EvidenceItem] = []


class AuthenticityReport(BaseModel):
    """The full report returned to the recruiter dashboard."""

    verdict: str  # short human verdict
    recommendation: str  # what to DO — always "review", never "auto-reject"
    overall_flag_score: float  # 0..1 combined concern level
    content: ContentAnalysis
    consistency: ConsistencyAnalysis
    behavior: BehaviorAnalysis
    evidence: List[EvidenceItem]  # merged + sorted by severity
    limitations: str  # honest note shown in the UI
    processing_ms: float
