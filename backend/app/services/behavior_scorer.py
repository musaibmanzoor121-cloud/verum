"""
services/behavior_scorer.py
===========================
ENGINE B (the interesting half) — "Was this submitted by a HUMAN or a SCRIPT?"

Naive bot checks (timing, headers) are trivial for a script to fake: just
`sleep(4)` and spoof a Chrome user-agent. So we ALSO score *behavioral
biometrics* captured while the form is filled in:

  * Mouse-movement entropy — humans move the cursor in irregular, curved paths;
    scripts either don't move it at all or move in perfectly straight lines.
  * Keystroke rhythm — humans have variable inter-key timing; scripts often
    inject text instantly or at a fixed cadence.

Plus the classic-but-still-useful signals:
  * Honeypot field — a hidden input real users never see; if it's filled,
    it was almost certainly a bot crawling the DOM.
  * Time-to-submit — implausibly fast completion.
  * Suspicious user-agent — headless browsers / HTTP libraries.

IMPORTANT (ethics + accessibility): this score NEVER auto-rejects. Behavioral
biometrics can misfire for users of assistive tech (screen readers, switch
access, keyboard-only navigation). It only raises a flag for HUMAN review.
"""

from __future__ import annotations

import math
from typing import List, Optional

from app.config import settings
from app.schemas import BehaviorAnalysis, BehaviorPayload, EvidenceItem

# User-agent fragments that indicate automation rather than a real browser.
BOT_UA_MARKERS = [
    "headless", "puppeteer", "playwright", "phantom", "selenium",
    "python-requests", "curl", "wget", "scrapy", "bot", "spider", "httpclient",
]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _normalized_entropy(values: List[float], bins: int = 8) -> float:
    """Shannon entropy of `values` binned into `bins`, normalized to 0..1.

    0 => all values identical (robotic).  1 => perfectly spread (human-like).
    Pure-Python so we don't hard-depend on numpy.
    """
    if len(values) < 2:
        return 0.0
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return 0.0
    counts = [0] * bins
    width = (hi - lo) / bins
    for v in values:
        idx = min(bins - 1, int((v - lo) / width))
        counts[idx] += 1
    total = sum(counts)
    entropy = 0.0
    for c in counts:
        if c:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy / math.log2(bins)


def _mouse_entropy(payload: BehaviorPayload) -> Optional[float]:
    """Entropy of movement DIRECTIONS between sampled mouse points."""
    pts = payload.mouse_events
    if len(pts) < 5:
        return None  # not enough movement to judge
    angles: List[float] = []
    for a, b in zip(pts, pts[1:]):
        dx, dy = b.x - a.x, b.y - a.y
        if dx == 0 and dy == 0:
            continue
        angles.append(math.atan2(dy, dx))  # -pi..pi
    return _normalized_entropy(angles, bins=8) if len(angles) >= 4 else None


def _keystroke_regularity(payload: BehaviorPayload) -> Optional[float]:
    """How ROBOTIC the typing rhythm is (1 = perfectly even, 0 = human-varied)."""
    gaps = [g for g in payload.key_intervals_ms if 0 < g < 5000]  # drop long pauses
    if len(gaps) < 6:
        return None
    mean = sum(gaps) / len(gaps)
    if mean <= 0:
        return None
    var = sum((g - mean) ** 2 for g in gaps) / len(gaps)
    cv = math.sqrt(var) / mean  # coefficient of variation
    # Low CV => very regular => robotic. Map CV=0 -> 1.0, CV>=0.35 -> 0.0
    return _clamp((0.35 - cv) / 0.35)


def score_behavior(
    payload: Optional[BehaviorPayload],
    user_agent: str = "",
    remote_ip: str = "",
) -> BehaviorAnalysis:
    """Compute a bot-likelihood report from behavioral + request signals."""
    evidence: List[EvidenceItem] = []

    # If the applicant form didn't send behavior data at all, we can only judge
    # the user-agent. Stay neutral rather than punishing missing data.
    honeypot_triggered = bool(payload and payload.honeypot.strip())
    time_to_submit: Optional[float] = None
    if payload:
        time_to_submit = max(0.0, (payload.submit_time - payload.form_load_time) / 1000.0)

    # ---- Signal: honeypot (strongest) -----------------------------------
    honeypot_score = 0.0
    if honeypot_triggered:
        honeypot_score = 0.97
        evidence.append(EvidenceItem(
            engine="behavior", signal="honeypot_filled",
            detail="A hidden honeypot field (invisible to humans) was filled in — "
                   "this is almost always an automated form-filler.",
            severity="high", excerpt=None,
        ))

    # ---- Signal: time-to-submit -----------------------------------------
    timing_sub = 0.0
    if time_to_submit is not None and time_to_submit < settings.MIN_SUBMIT_SECONDS:
        timing_sub = _clamp(1.0 - (time_to_submit / settings.MIN_SUBMIT_SECONDS))
        evidence.append(EvidenceItem(
            engine="behavior", signal="fast_submit",
            detail=f"Form submitted in {time_to_submit:.1f}s — faster than a human "
                   f"could plausibly read and complete it (threshold "
                   f"{settings.MIN_SUBMIT_SECONDS:.0f}s).",
            severity="medium" if timing_sub < 0.7 else "high", excerpt=None,
        ))

    # ---- Signal: user-agent ---------------------------------------------
    ua_low = (user_agent or "").lower()
    suspicious_ua = (not ua_low) or any(m in ua_low for m in BOT_UA_MARKERS)
    ua_sub = 1.0 if suspicious_ua else 0.0
    if suspicious_ua:
        evidence.append(EvidenceItem(
            engine="behavior", signal="suspicious_user_agent",
            detail=f"Request came from a suspicious or missing user-agent "
                   f"('{user_agent or 'empty'}'), typical of scripts/headless browsers.",
            severity="medium", excerpt=None,
        ))

    # ---- Signal: mouse entropy ------------------------------------------
    mouse_entropy = _mouse_entropy(payload) if payload else None
    mouse_sub = 0.0
    if mouse_entropy is not None:
        mouse_sub = _clamp(1.0 - mouse_entropy)  # low entropy => bot-like
        if mouse_sub > 0.6:
            evidence.append(EvidenceItem(
                engine="behavior", signal="low_mouse_entropy",
                detail=f"Mouse movement was unusually uniform (entropy "
                       f"{mouse_entropy:.2f}); human cursor paths are typically more varied.",
                severity="low", excerpt=None,
            ))

    # ---- Signal: keystroke regularity -----------------------------------
    keystroke_regularity = _keystroke_regularity(payload) if payload else None
    keys_sub = 0.0
    if keystroke_regularity is not None:
        keys_sub = keystroke_regularity
        if keys_sub > 0.6:
            evidence.append(EvidenceItem(
                engine="behavior", signal="robotic_typing",
                detail=f"Typing rhythm was almost perfectly even "
                       f"(regularity {keystroke_regularity:.2f}); humans type with "
                       f"variable inter-key timing.",
                severity="medium", excerpt=None,
            ))

    # ---- Combine ---------------------------------------------------------
    # Most signals blend together with weights...
    weighted = (0.35 * timing_sub + 0.30 * ua_sub
                + 0.20 * keys_sub + 0.15 * mouse_sub)
    # ...but a few signals are strong enough to stand on their own. A filled
    # honeypot or a sub-second submit is almost never a real human, so the
    # final score is the MAX of the blend and any individually-strong signal.
    strong_signals = [honeypot_score]
    if time_to_submit is not None and time_to_submit < 1.0:
        strong_signals.append(0.70)
    score = _clamp(max(max(strong_signals), weighted))

    return BehaviorAnalysis(
        bot_likelihood_band=_score_to_band(score),
        bot_likelihood_score=round(score, 3),
        honeypot_triggered=honeypot_triggered,
        time_to_submit_seconds=round(time_to_submit, 2) if time_to_submit is not None else None,
        suspicious_user_agent=suspicious_ua,
        mouse_entropy=round(mouse_entropy, 3) if mouse_entropy is not None else None,
        keystroke_regularity=round(keystroke_regularity, 3) if keystroke_regularity is not None else None,
        evidence=evidence,
    )


def _score_to_band(score: float) -> str:
    center = round(score * 100)
    lo, hi = max(0, center - 10), min(100, center + 10)
    return f"{lo}–{hi}% likely automated"
