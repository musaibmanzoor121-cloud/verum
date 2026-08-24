"""
services/ai_analyzer.py
=======================
ENGINE A — "Is this writing likely AI-generated?"

We combine several independent, cheap statistical signals. No single one is
proof; together they form a confidence BAND. Every signal also produces
EVIDENCE (the exact phrases / concrete numbers) so a recruiter sees *why*.

The three core ideas
--------------------
1. PERPLEXITY  — how "surprised" a language model is by the text.
   AI text is optimized to be probable, so it tends to be LOW-perplexity
   (predictable). Human text is messier -> higher perplexity.
   * If USE_TRANSFORMER_PERPLEXITY=true and torch/transformers are installed,
     we compute REAL perplexity with GPT-2.
   * Otherwise we skip it and lean on the other signals (graceful degradation).
     This keeps the app fast and deployable on a 512MB free tier.

2. BURSTINESS — variance in sentence length. Humans write uneven sentences
   (a 4-word punch, then a 30-word ramble). AI writing is eerily uniform.
   We measure the coefficient of variation (std / mean) of sentence lengths.

3. BUZZWORD DENSITY — generic AI/HR filler ("spearheaded cross-functional
   initiatives", "leveraged cutting-edge solutions"). We count hits per 100
   words and remember exactly which phrases fired.

We also track REPEATED SENTENCE OPENERS ("Moreover, ... Furthermore, ...")
which are a classic tell of templated AI prose.

The final score (0..1, higher = more AI-like) is a weighted blend, then we
express it as a human-friendly band like "60–80% likely AI-assisted".
"""

from __future__ import annotations

import re
import statistics
from functools import lru_cache
from typing import List, Optional

from app.config import settings
from app.schemas import ContentAnalysis, EvidenceItem

# ---------------------------------------------------------------------------
# Word lists (tunable). Keeping them here makes the logic easy to explain.
# ---------------------------------------------------------------------------
BUZZWORDS = [
    "spearheaded", "cross-functional", "leveraged", "leverage", "cutting-edge",
    "synergy", "results-driven", "detail-oriented", "team player", "go-getter",
    "think outside the box", "dynamic", "proactive", "passionate about",
    "proven track record", "value add", "deep dive", "move the needle",
    "best-in-class", "seamless", "seamlessly", "robust solutions",
    "innovative solutions", "state-of-the-art", "paradigm", "holistic",
    "strategic initiatives", "core competencies", "utilize", "utilized",
    "fast-paced", "ever-evolving", "delve", "tapestry", "underscore",
    "pivotal", "commendable", "meticulous", "realm", "landscape",
    "in today's world", "furthermore", "moreover", "cutting edge",
]

# Words that AI loves to START sentences with. Heavy repetition = templated.
TRANSITION_OPENERS = {
    "additionally", "moreover", "furthermore", "however", "therefore",
    "consequently", "in conclusion", "overall", "firstly", "secondly",
    "finally", "notably", "importantly", "ultimately", "thus",
}


# ---------------------------------------------------------------------------
# Small text utilities
# ---------------------------------------------------------------------------
def split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter (no NLTK download needed).

    Splits on ., !, ? followed by whitespace. Good enough for scoring and
    avoids the deployment headache of downloading NLTK's 'punkt' data.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z']+", text))


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Optional: real transformer perplexity (only if explicitly enabled)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_model():
    """Load GPT-2 once and cache it. Called only if the transformer path is on."""
    import torch  # noqa: F401  (heavy import, hence lazy)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(settings.PERPLEXITY_MODEL)
    model = AutoModelForCausalLM.from_pretrained(settings.PERPLEXITY_MODEL)
    model.eval()
    return tok, model


def compute_transformer_perplexity(text: str) -> Optional[float]:
    """Return GPT-2 perplexity, or None if anything goes wrong."""
    try:
        import torch

        tok, model = _load_model()
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"]
        with torch.no_grad():
            out = model(input_ids, labels=input_ids)
        # out.loss is average negative log-likelihood; perplexity = exp(loss).
        return float(torch.exp(out.loss).item())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Individual signal calculators
# ---------------------------------------------------------------------------
def burstiness_score(sentences: List[str]) -> float:
    """Coefficient of variation of sentence word-lengths.

    ~0.0  => every sentence the same length (robotic)
    ~0.5+ => healthy human variation
    """
    lengths = [word_count(s) for s in sentences if word_count(s) >= 3]
    if len(lengths) < 2:
        return 0.5  # not enough data; stay neutral
    mean = statistics.mean(lengths)
    if mean == 0:
        return 0.0
    std = statistics.pstdev(lengths)
    return std / mean


def find_buzzwords(text: str) -> List[str]:
    """Return the list of buzzword phrases actually present (with counts)."""
    low = text.lower()
    found: List[str] = []
    for phrase in BUZZWORDS:
        n = low.count(phrase.lower())
        if n > 0:
            found.append(phrase if n == 1 else f"{phrase} (x{n})")
    return found


def repeated_openers(sentences: List[str]) -> List[str]:
    """Return transition words that are over-used as sentence openers."""
    openers: dict[str, int] = {}
    for s in sentences:
        first = re.findall(r"[A-Za-z]+", s)
        if not first:
            continue
        w = first[0].lower()
        if w in TRANSITION_OPENERS:
            openers[w] = openers.get(w, 0) + 1
    # Flag any opener used 2+ times.
    return [f"{w} (x{c})" for w, c in openers.items() if c >= 2]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def analyze_content(text: str, metadata_flags: Optional[List[str]] = None) -> ContentAnalysis:
    """Run Engine A on a document and return a fully-populated ContentAnalysis."""
    metadata_flags = metadata_flags or []
    evidence: List[EvidenceItem] = []

    sentences = split_sentences(text)
    words = word_count(text)

    # ---- Signal 1: burstiness -------------------------------------------
    burst = burstiness_score(sentences)
    # Convert to "AI-ness": low variation => high AI likelihood.
    ai_from_burst = _clamp((0.55 - burst) / 0.55)
    if ai_from_burst > 0.5 and len(sentences) >= 4:
        evidence.append(EvidenceItem(
            engine="content",
            signal="low_burstiness",
            detail=(f"Sentence lengths are unusually uniform "
                    f"(variation index {burst:.2f}; human writing is typically >0.45). "
                    f"Uniform rhythm is a common trait of AI-generated text."),
            severity="medium" if ai_from_burst < 0.75 else "high",
            excerpt=sentences[0][:160] if sentences else None,
        ))

    # ---- Signal 2: buzzword density -------------------------------------
    buzzwords = find_buzzwords(text)
    density = (len(buzzwords) / max(words, 1)) * 100  # hits per 100 words
    ai_from_buzz = _clamp(density / 4.0)
    if buzzwords:
        evidence.append(EvidenceItem(
            engine="content",
            signal="buzzword_density",
            detail=(f"Found {len(buzzwords)} generic buzzword/filler phrase(s) "
                    f"({density:.1f} per 100 words). Heavy filler is typical of "
                    f"AI-generated or template resumes."),
            severity="low" if density < 2 else ("medium" if density < 4 else "high"),
            excerpt=", ".join(buzzwords[:8]),
        ))

    # ---- Signal 3: repeated transition openers --------------------------
    openers = repeated_openers(sentences)
    ai_from_openers = _clamp(len(openers) / 3.0)
    if openers:
        evidence.append(EvidenceItem(
            engine="content",
            signal="repeated_openers",
            detail=("Multiple sentences start with the same transition word "
                    "(a hallmark of templated AI prose)."),
            severity="low" if len(openers) < 2 else "medium",
            excerpt="; ".join(openers),
        ))

    # ---- Signal 4: perplexity (optional transformer path) ---------------
    perplexity: Optional[float] = None
    perplexity_method = "heuristic"
    ai_from_ppl = None
    if settings.USE_TRANSFORMER_PERPLEXITY:
        perplexity = compute_transformer_perplexity(text)
        if perplexity is not None:
            perplexity_method = "gpt2"
            # Low perplexity => predictable => more AI-like.
            ai_from_ppl = _clamp((70.0 - perplexity) / 60.0)
            evidence.append(EvidenceItem(
                engine="content",
                signal="low_perplexity",
                detail=(f"GPT-2 perplexity is {perplexity:.1f}. Lower values mean the "
                        f"text is highly predictable to a language model, which "
                        f"correlates with AI generation."),
                severity="low" if ai_from_ppl < 0.4 else ("medium" if ai_from_ppl < 0.7 else "high"),
                excerpt=None,
            ))

    # ---- Metadata flags become content evidence -------------------------
    for f in metadata_flags:
        evidence.append(EvidenceItem(
            engine="metadata", signal="document_metadata",
            detail=f, severity="medium", excerpt=None,
        ))

    # ---- Combine into one score -----------------------------------------
    # Weights differ depending on whether we have a real perplexity number.
    if ai_from_ppl is not None:
        score = (0.40 * ai_from_ppl + 0.30 * ai_from_burst
                 + 0.20 * ai_from_buzz + 0.10 * ai_from_openers)
    else:
        score = (0.50 * ai_from_burst + 0.35 * ai_from_buzz
                 + 0.15 * ai_from_openers)
    # A little nudge up if document metadata itself looks automated.
    if metadata_flags:
        score = _clamp(score + 0.10)
    score = _clamp(score)

    return ContentAnalysis(
        ai_likelihood_band=_score_to_band(score, "likely AI-assisted"),
        ai_likelihood_score=round(score, 3),
        perplexity=round(perplexity, 2) if perplexity is not None else None,
        perplexity_method=perplexity_method,
        burstiness=round(burst, 3),
        buzzword_density=round(density, 2),
        buzzwords_found=buzzwords,
        metadata_flags=metadata_flags,
        evidence=evidence,
    )


def _score_to_band(score: float, suffix: str) -> str:
    """Turn a 0..1 score into an honest range string, not a fake-precise number.

    Example: 0.68 -> "58–78% likely AI-assisted"
    """
    center = round(score * 100)
    lo = max(0, center - 10)
    hi = min(100, center + 10)
    return f"{lo}–{hi}% {suffix}"
