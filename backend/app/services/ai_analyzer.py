"""
services/ai_analyzer.py
=======================
ENGINE A — "Is this writing likely AI-generated?"

We combine several independent, cheap statistical + lexical signals. No single
one is proof; together they form a confidence BAND. Every signal also produces
EVIDENCE (the exact phrases / concrete numbers) so a recruiter sees *why*.

The signals
-----------
1. BURSTINESS — variance in sentence length. Older models wrote eerily uniform
   sentences. Modern frontier models (2025+) deliberately vary sentence length,
   so this signal is now the WEAKEST tell and carries a small weight.

2. BUZZWORD / FILLER DENSITY — generic HR/AI filler ("results-driven",
   "leveraged", "passionate", "spearheaded") per 100 words.

3. AI CADENCE PHRASES — stock phrasings that current chat models overuse in
   resumes and cover letters ("a testament to", "thrive in fast-paced
   environments", "look forward to the opportunity", "not only ... but also").
   These are the STRONGEST cheap tell against a modern LLM.

4. EM-DASH CHARACTER — real applicants type a hyphen "-"; chat models emit the
   typographic em-dash "—" (and en-dash "–"). Its presence is a quiet giveaway.

5. TRIADIC "RULE OF THREE" LISTS — "design, build, and deploy" style triples.
   Chat models love parallel triads far more than most human writers.

6. REPEATED SENTENCE OPENERS — "Moreover, ... Furthermore, ...".

7. PERPLEXITY (optional) — real GPT-2 surprise score, only if
   USE_TRANSFORMER_PERPLEXITY=true (heavy; off by default for free-tier deploys).

The final score (0..1, higher = more AI-like) is a weighted blend, then we
express it as a human-friendly band like "60–80% likely AI-assisted".

IMPORTANT (fairness): the thresholds are tuned so that specific, concrete,
uneven *human* writing stays low. We would rather miss some AI text than
wrongly flag a real person — flags only ever route to a human, never reject.
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
# Single words / short phrases that read as generic AI/HR filler.
BUZZWORDS = [
    "spearheaded", "spearhead", "cross-functional", "leveraged", "leverage",
    "cutting-edge", "cutting edge", "synergy", "results-driven",
    "results-oriented", "detail-oriented", "detail oriented", "team player",
    "go-getter", "self-starter", "think outside the box", "dynamic",
    "proactive", "passionate", "passion", "dedicated", "driven",
    "proven track record", "value add", "value-add", "deep dive",
    "move the needle", "best-in-class", "seamless", "seamlessly",
    "robust", "robust solutions", "innovative", "innovative solutions",
    "innovation", "state-of-the-art", "paradigm", "holistic", "scalable",
    "strategic initiatives", "core competencies", "utilize", "utilized",
    "fast-paced", "ever-evolving", "delve", "tapestry", "underscore",
    "pivotal", "commendable", "meticulous", "realm", "landscape",
    "in today's world", "furthermore", "moreover", "meaningful", "impactful",
    "streamlined", "streamline", "honed", "thrive", "eager", "foster",
    "cultivate", "empower", "elevate", "harness", "myriad", "plethora",
    "comprehensive", "adept", "poised", "vibrant", "keen",
]

# Words that AI loves to START sentences with. Heavy repetition = templated.
TRANSITION_OPENERS = {
    "additionally", "moreover", "furthermore", "however", "therefore",
    "consequently", "in conclusion", "overall", "firstly", "secondly",
    "finally", "notably", "importantly", "ultimately", "thus",
}

# Multi-word phrasings that current chat models overuse. Case-insensitive.
# These are the single most useful cheap signal against a modern LLM.
AI_CADENCE_PATTERNS = [
    r"a testament to",
    r"plays? a (?:crucial|vital|pivotal|key|significant|central|major) role",
    r"navigat\w+ the (?:complex|complexit\w+|landscape|challenges|intricac\w+)",
    r"in (?:todays?|the) (?:ever-evolving|fast-paced|dynamic|modern|competitive|digital|rapidly changing) (?:world|landscape|environment|era|market|field)",
    r"when it comes to",
    r"it(?:'s| is) worth noting",
    r"not only\b.{0,80}?\bbut also",
    r"at the end of the day",
    r"underscor\w+ the importance",
    r"highlight\w* the importance",
    r"a (?:wide range|myriad|plethora|wealth|multitude|diverse array) of",
    r"meticulous attention to detail",
    r"proven (?:ability|track record)",
    r"passion(?:ate)? for",
    r"look(?:ing)? forward to (?:the )?(?:opportunity|possibility|chance|contributing)",
    r"make (?:a )?(?:meaningful|significant|lasting|real|positive) (?:impact|contribution|difference)",
    r"thrive in\b",
    r"unique (?:blend|combination) of",
    r"valuable (?:addition|asset) to",
    r"seamlessly (?:integrat\w+|blend\w*|combin\w+)",
    r"delve into",
    r"eager to (?:contribute|learn|apply|join|leverage)",
    r"committed to (?:excellence|delivering|providing|continuous)",
    r"honed my (?:skills|abilit\w+|craft|expertise)",
    r"wealth of (?:experience|knowledge)",
    r"align\w* (?:with|perfectly with) your (?:mission|vision|values|goals|needs)",
    r"drawn to the (?:intersection|opportunity|challenge)",
    r"blend of technical (?:expertise|skills)",
    r"strong (?:foundation|background) in",
    r"contribute to (?:your|the) (?:team|organization|company|mission)('s)?",
]
_CADENCE_RE = [re.compile(p, re.IGNORECASE) for p in AI_CADENCE_PATTERNS]

# Typographic dashes a human almost never types by hand (they use "-").
EM_DASH_CHARS = ("—", "–")  # — (em dash), – (en dash)

# "word, word, and word" parallel triads (the rule-of-three chat models love).
_TRIADIC_RE = re.compile(
    r"\b[\w-]+,\s+[\w-]+,\s+(?:and|or)\s+[\w-]+", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Small text utilities
# ---------------------------------------------------------------------------
def split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter (no NLTK download needed)."""
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
    """Return the list of buzzword phrases actually present (with counts).

    Uses word-boundary matching so short words (e.g. 'passion', 'driven') don't
    match inside unrelated words.
    """
    low = text.lower()
    found: List[str] = []
    for phrase in BUZZWORDS:
        pat = r"\b" + re.escape(phrase.lower()) + r"\b"
        n = len(re.findall(pat, low))
        if n > 0:
            found.append(phrase if n == 1 else f"{phrase} (x{n})")
    return found


def find_cadence_phrases(text: str) -> List[str]:
    """Return the actual AI-cadence phrases found in the text (for evidence)."""
    hits: List[str] = []
    for rx in _CADENCE_RE:
        for m in rx.finditer(text):
            snippet = re.sub(r"\s+", " ", m.group(0)).strip()
            if snippet and snippet.lower() not in [h.lower() for h in hits]:
                hits.append(snippet)
    return hits


def count_em_dashes(text: str) -> int:
    return sum(text.count(ch) for ch in EM_DASH_CHARS)


def find_triads(text: str) -> List[str]:
    return [re.sub(r"\s+", " ", m.group(0)).strip() for m in _TRIADIC_RE.finditer(text)]


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
    return [f"{w} (x{c})" for w, c in openers.items() if c >= 2]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def analyze_content(text: str, metadata_flags: Optional[List[str]] = None) -> ContentAnalysis:
    """Run Engine A on a document (or résumé + cover letter combined)."""
    metadata_flags = metadata_flags or []
    evidence: List[EvidenceItem] = []

    sentences = split_sentences(text)
    words = word_count(text)

    # ---- Signal 1: burstiness (weak vs modern LLMs — small weight) -------
    burst = burstiness_score(sentences)
    ai_from_burst = _clamp((0.45 - burst) / 0.45)
    if ai_from_burst > 0.55 and len(sentences) >= 4:
        evidence.append(EvidenceItem(
            engine="content",
            signal="low_burstiness",
            detail=(f"Sentence lengths are unusually uniform "
                    f"(variation index {burst:.2f}; varied human writing is typically >0.45)."),
            severity="low" if ai_from_burst < 0.8 else "medium",
            excerpt=sentences[0][:160] if sentences else None,
        ))

    # ---- Signal 2: buzzword / filler density -----------------------------
    buzzwords = find_buzzwords(text)
    buzz_hits = sum(int(re.search(r"\(x(\d+)\)", b).group(1)) if "(x" in b else 1
                    for b in buzzwords)
    density = (buzz_hits / max(words, 1)) * 100  # hits per 100 words
    ai_from_buzz = _clamp(density / 2.0)
    if buzzwords:
        evidence.append(EvidenceItem(
            engine="content",
            signal="buzzword_density",
            detail=(f"Found {buzz_hits} generic buzzword/filler term(s) "
                    f"({density:.1f} per 100 words). Heavy filler is typical of "
                    f"AI-generated or template writing."),
            severity="low" if density < 1.5 else ("medium" if density < 3 else "high"),
            excerpt=", ".join(buzzwords[:10]),
        ))

    # ---- Signal 3: AI cadence phrases (strongest modern tell) ------------
    cadence = find_cadence_phrases(text)
    ai_from_cadence = _clamp(len(cadence) / 4.0)
    if cadence:
        evidence.append(EvidenceItem(
            engine="content",
            signal="ai_cadence_phrases",
            detail=(f"Found {len(cadence)} stock phrasing(s) that current AI writing "
                    f"assistants heavily overuse. One or two can be coincidence; "
                    f"several together is a strong tell."),
            severity="low" if len(cadence) < 2 else ("medium" if len(cadence) < 4 else "high"),
            excerpt="; ".join(cadence[:6]),
        ))

    # ---- Signal 4: typographic em-dash character -------------------------
    em_dashes = count_em_dashes(text)
    ai_from_emdash = _clamp(em_dashes / 3.0)
    if em_dashes >= 1:
        evidence.append(EvidenceItem(
            engine="content",
            signal="em_dash_usage",
            detail=(f"Text uses the typographic em/en-dash character {em_dashes} time(s). "
                    f"People typing by hand almost always use a plain hyphen '-'; the "
                    f"'—' character is a common trait of machine-generated text."),
            severity="low" if em_dashes < 3 else "medium",
            excerpt=None,
        ))

    # ---- Signal 5: triadic "rule of three" lists -------------------------
    triads = find_triads(text)
    ai_from_triadic = _clamp(len(triads) / 3.0)
    if len(triads) >= 2:
        evidence.append(EvidenceItem(
            engine="content",
            signal="triadic_lists",
            detail=(f"Found {len(triads)} parallel 'X, Y, and Z' triad(s). AI writing "
                    f"leans on the rule-of-three far more often than most human writers."),
            severity="low" if len(triads) < 3 else "medium",
            excerpt="; ".join(triads[:4]),
        ))

    # ---- Signal 6: repeated transition openers ---------------------------
    openers = repeated_openers(sentences)
    ai_from_openers = _clamp(len(openers) / 2.0)
    if openers:
        evidence.append(EvidenceItem(
            engine="content",
            signal="repeated_openers",
            detail=("Multiple sentences start with the same transition word "
                    "(a hallmark of templated AI prose)."),
            severity="low" if len(openers) < 2 else "medium",
            excerpt="; ".join(openers),
        ))

    # ---- Signal 7: perplexity (optional transformer path) ----------------
    perplexity: Optional[float] = None
    perplexity_method = "heuristic"
    ai_from_ppl = None
    if settings.USE_TRANSFORMER_PERPLEXITY:
        perplexity = compute_transformer_perplexity(text)
        if perplexity is not None:
            perplexity_method = "gpt2"
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
    # Lexical + cadence signals do the heavy lifting; burstiness is a minor
    # tie-breaker because modern models defeat it. When a real GPT-2 perplexity
    # number is available, it joins the blend.
    if ai_from_ppl is not None:
        score = (0.22 * ai_from_ppl + 0.20 * ai_from_buzz + 0.20 * ai_from_cadence
                 + 0.15 * ai_from_emdash + 0.14 * ai_from_triadic
                 + 0.05 * ai_from_openers + 0.04 * ai_from_burst)
    else:
        score = (0.28 * ai_from_buzz + 0.24 * ai_from_cadence
                 + 0.16 * ai_from_emdash + 0.15 * ai_from_triadic
                 + 0.09 * ai_from_openers + 0.08 * ai_from_burst)

    # ---- Corroboration boost --------------------------------------------
    # Several INDEPENDENT tells firing together is far stronger evidence than
    # one tell firing hard. A modern chat-model application typically trips 3-4
    # different signals at once (em-dash + triads + cadence + filler) even when
    # each is individually mild — while a genuine person typing by hand trips
    # zero or one. Crucially, this boost keys off the *count of distinct
    # signals*, not their strength, so it CANNOT be triggered by a single
    # stylistic quirk: an honest writer who happens to use one em-dash, or a few
    # buzzwords, lights up one signal and gets ZERO boost. It only bites when
    # multiple independent machine-tells co-occur — the pattern real applicants
    # don't reproduce. We make it decisive (not marginal) so multi-signal
    # AI-polished text lands clearly inside human review rather than hovering on
    # the threshold, while every 0-1 signal writer is left completely untouched.
    distinct_signals = len({e.signal for e in evidence if e.engine == "content"})
    corroboration = min(0.34, 0.14 * max(0, distinct_signals - 1))
    score += corroboration

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
    """Turn a 0..1 score into an honest range string, not a fake-precise number."""
    center = round(score * 100)
    lo = max(0, center - 10)
    hi = min(100, center + 10)
    return f"{lo}–{hi}% {suffix}"
