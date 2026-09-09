"""
services/features.py
====================
Turn a piece of text into a fixed NUMERIC FEATURE VECTOR for the learned model.

This is the bridge between Engine A's hand-written signal detectors and the
trained logistic-regression classifier (see services/ml_model.py and
eval/train_model.py). It deliberately reuses the *exact* signal functions the
live heuristic engine uses (`app.services.ai_analyzer`), so the model is trained
and served on the same numbers the rule-based engine already computes — there is
no separate, drift-prone feature pipeline.

The 7 features (every one is already in the range 0..1)
-------------------------------------------------------
Each of the first six is one of Engine A's per-signal sub-scores — the same
normalized 0..1 quantities the heuristic blends with hand-picked weights. The
whole point of the ML upgrade is to *learn* those weights from labeled data
instead of guessing them, so the learned coefficients are directly comparable to
the heuristic's constants.

    buzzword          filler density        (clamp(density / 2.0))
    cadence           stock AI phrasings     (clamp(#phrases / 4.0))
    em_dash           typographic dashes     (clamp(#dashes / 3.0))
    triadic           rule-of-three lists    (clamp(#triads / 3.0))
    openers           repeated openers       (clamp(#repeated / 2.0))
    burstiness        sentence uniformity    (clamp((0.45 - CoV) / 0.45))

The seventh feature captures *corroboration* — how many independent tells fire
at once — which is the single most important idea in the heuristic (several mild
signals together beat one strong signal). It uses the SAME gating thresholds the
engine uses to decide whether a signal is worth showing as evidence:

    distinct_signals  # of the 6 signals that fired / 6.0

Keeping the feature set small and interpretable is intentional: with a linear
model, each coefficient reads as "how much this tell moves the AI score," which
is exactly the explainability the product promises.
"""

from __future__ import annotations

from typing import Dict, List

from app.services.ai_analyzer import (
    _clamp,
    burstiness_score,
    count_buzzword_hits,
    count_em_dashes,
    find_buzzwords,
    find_cadence_phrases,
    find_triads,
    repeated_openers,
    split_sentences,
    word_count,
)

# The canonical feature order. model.json stores this so serving and training
# can never disagree about which column is which.
FEATURE_NAMES: List[str] = [
    "buzzword",
    "cadence",
    "em_dash",
    "triadic",
    "openers",
    "burstiness",
    "distinct_signals",
]

# Number of base signals that can "fire" (everything except the corroboration
# feature itself). Used to normalize the distinct-signal count into 0..1.
_N_BASE_SIGNALS = 6


def extract_features(text: str) -> Dict[str, float]:
    """Compute the 7 model features for a document, as a name->value dict.

    Mirrors `ai_analyzer.analyze_content` exactly: the same sub-scores and the
    same evidence-gating thresholds that decide whether a signal counts toward
    corroboration.
    """
    sentences = split_sentences(text)
    words = word_count(text)

    # --- the six base sub-scores (identical to analyze_content) ------------
    burst = burstiness_score(sentences)
    f_burst = _clamp((0.45 - burst) / 0.45)

    buzzwords = find_buzzwords(text)
    buzz_hits = count_buzzword_hits(found=buzzwords)
    density = (buzz_hits / max(words, 1)) * 100
    f_buzz = _clamp(density / 2.0)

    cadence = find_cadence_phrases(text)
    f_cadence = _clamp(len(cadence) / 4.0)

    em_dashes = count_em_dashes(text)
    f_emdash = _clamp(em_dashes / 3.0)

    triads = find_triads(text)
    f_triadic = _clamp(len(triads) / 3.0)

    openers = repeated_openers(sentences)
    f_openers = _clamp(len(openers) / 2.0)

    # --- corroboration: how many signals FIRED (same gates as the engine) --
    # These conditions are copied verbatim from analyze_content's evidence
    # gating so the model sees the same "distinct signal" count a recruiter
    # would see as evidence rows.
    fired = 0
    if f_burst > 0.55 and len(sentences) >= 4:
        fired += 1
    if buzzwords:
        fired += 1
    if cadence:
        fired += 1
    if em_dashes >= 1:
        fired += 1
    if len(triads) >= 2:
        fired += 1
    if openers:
        fired += 1
    f_distinct = fired / _N_BASE_SIGNALS

    return {
        "buzzword": round(f_buzz, 6),
        "cadence": round(f_cadence, 6),
        "em_dash": round(f_emdash, 6),
        "triadic": round(f_triadic, 6),
        "openers": round(f_openers, 6),
        "burstiness": round(f_burst, 6),
        "distinct_signals": round(f_distinct, 6),
    }


def extract_vector(text: str) -> List[float]:
    """Same as `extract_features`, but as a plain list in `FEATURE_NAMES` order."""
    feats = extract_features(text)
    return [feats[name] for name in FEATURE_NAMES]
