"""
eval/robustness.py
==================
How easily can a determined applicant DEFEAT Engine A?

This is the honest counterpart to eval/evaluate.py. Every stylometric AI-text
detector on earth can be evaded by lightly editing the draft, and a portfolio
project that pretends otherwise is not credible. So instead of hiding that
weakness, we MEASURE it: we take the AI samples, apply realistic "humanizing"
edits, and report exactly how far the score drops and how many samples slip
back under the product's review line.

The edits (each is something a real person actually does)
---------------------------------------------------------
1. de-dash          — swap the typographic "—"/"–" for a plain hyphen "-"
2. drop Oxford comma — "design, build, and ship" -> "design, build and ship"
                       (kills the triad detector, changes no words)
3. soften cadence   — replace stock AI phrasings with plain wording
4. plain buzzwords  — replace corporate filler with everyday words
5. full humanize    — all of the above at once (the realistic adversary)

None of these change the *meaning* of the text — they only remove the surface
tells. That is the whole point: surface tells are cheap to remove.

WHAT THIS PROVES (and why it is a strength, not an embarrassment)
-----------------------------------------------------------------
It shows we understand the limits of the method. Verum's design answers this
weakness in two ways the report makes explicit:
  * it NEVER auto-rejects — every flag is a routing decision, so a miss is
    recoverable; and
  * content is only one of three engines. Editing the prose does nothing to the
    submission-behavior signals (timing, mouse, keystroke rhythm, honeypot),
    which an evader would also have to fake.

HOW TO RUN (from the backend/ folder, in your normal venv):

    cd backend
    python -m eval.robustness      # prints the table and writes eval/ROBUSTNESS.md
"""

from __future__ import annotations

import os
import re
import sys
from typing import Callable, List, Tuple

# Make `app` importable whether run from repo root or backend/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from app.services.ai_analyzer import analyze_content  # noqa: E402

from eval.dataset import SAMPLES  # noqa: E402
from eval.evaluate import PRODUCT_THRESHOLD  # noqa: E402  (single source of truth)


# ---------------------------------------------------------------------------
# The humanizing edits
# ---------------------------------------------------------------------------
_EM_DASH_CHARS = ("—", "–")


def strip_em_dashes(text: str) -> str:
    """Replace the typographic em/en-dash with a plain hyphen a human would type."""
    for ch in _EM_DASH_CHARS:
        text = text.replace(ch, "-")
    return text


def drop_oxford_comma(text: str) -> str:
    """Turn 'a, b, and c' into 'a, b and c'.

    The triad detector needs *two* commas ('a, b, and c'); removing the comma
    before 'and'/'or' defeats it while changing not one word. Many style guides
    omit the Oxford comma, so this is a completely natural edit.
    """
    return re.sub(
        r"(\b[\w-]+,\s+[\w-]+)(,)(\s+(?:and|or)\s+[\w-]+)",
        r"\1\3",
        text,
    )


# Stock AI phrasings -> plain equivalents. Case-insensitive.
_CADENCE_REPLACEMENTS: List[Tuple[str, str]] = [
    (r"I am writing to express my (?:enthusiasm|interest)(?: for| in)?", "I want to apply for"),
    (r"look(?:ing)? forward to the opportunity to contribute to your continued success", "hope to help the team"),
    (r"look(?:ing)? forward to (?:the )?(?:opportunity|possibility|chance)[^.]*", "hope to talk"),
    (r"a testament to", "which shows"),
    (r"thrive in", "do well in"),
    (r"unique (?:blend|combination) of", "mix of"),
    (r"valuable (?:addition|asset) to", "good fit for"),
    (r"passionate about", "interested in"),
    (r"proven (?:track record|ability)", "solid experience"),
    (r"honed my (?:ability|skills|craft|expertise)", "gotten better at"),
    (r"drawn to the (?:intersection|opportunity|challenge)[^.]*", "interested in this work"),
    (r"in (?:today's|the) (?:fast-paced|ever-evolving|competitive|digital|dynamic|modern)[^.,]*", "these days"),
    (r"committed to (?:excellence|continuous learning|delivering)", "focused on the work"),
    (r"not only\b(.{0,80}?)\bbut also", r"\1and"),
    (r"make (?:a )?(?:meaningful|significant|lasting|real|positive) (?:impact|contribution|difference)", "help out"),
]


def soften_cadence(text: str) -> str:
    for pat, repl in _CADENCE_REPLACEMENTS:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text


# Corporate filler -> everyday words. Case-insensitive.
_BUZZWORD_REPLACEMENTS: List[Tuple[str, str]] = [
    (r"results-(?:driven|oriented)", "focused"),
    (r"detail-oriented", "careful"),
    (r"leverag(?:e|es|ed|ing)", "use"),
    (r"spearhead(?:s|ed|ing)?", "led"),
    (r"cutting-edge", "new"),
    (r"best-in-class", "good"),
    (r"robust", "solid"),
    (r"scalable", "large"),
    (r"innovative", "new"),
    (r"cross-functional", "cross-team"),
    (r"synergy", "teamwork"),
    (r"seamless(?:ly)?", "smooth"),
    (r"passionate", "keen"),
    (r"\bdynamic\b", "busy"),
    (r"utiliz(?:e|es|ed|ing)", "use"),
    (r"move the needle", "make progress"),
    (r"think outside the box", "come up with ideas"),
]


def plain_buzzwords(text: str) -> str:
    for pat, repl in _BUZZWORD_REPLACEMENTS:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text


def full_humanize(text: str) -> str:
    """The realistic adversary: apply every edit at once."""
    return plain_buzzwords(soften_cadence(drop_oxford_comma(strip_em_dashes(text))))


TRANSFORMS: List[Tuple[str, str, Callable[[str], str]]] = [
    ("de-dash", "swap “—”/“–” for a plain hyphen", strip_em_dashes),
    ("drop Oxford comma", "remove the comma before ‘and’ in triads", drop_oxford_comma),
    ("soften cadence", "swap stock AI phrasings for plain wording", soften_cadence),
    ("plain buzzwords", "replace corporate filler with everyday words", plain_buzzwords),
    ("full humanize", "all four edits applied together", full_humanize),
]


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def _score(text: str) -> float:
    return float(analyze_content(text).ai_likelihood_score)


def run():
    ai_samples = [s for s in SAMPLES if s["label"] == "ai"]
    base_scores = {s["id"]: _score(s["text"]) for s in ai_samples}
    n = len(ai_samples)
    base_mean = sum(base_scores.values()) / n
    base_evaded = sum(1 for v in base_scores.values() if v < PRODUCT_THRESHOLD)

    # Aggregate effect of each transform.
    rows = []
    for name, _desc, fn in TRANSFORMS:
        after = {s["id"]: _score(fn(s["text"])) for s in ai_samples}
        after_mean = sum(after.values()) / n
        evaded = sum(1 for v in after.values() if v < PRODUCT_THRESHOLD)
        drop = round(base_mean - after_mean, 3) + 0.0  # +0.0 normalizes -0.0 -> 0.0
        rows.append((name, base_mean, after_mean, drop, evaded))

    # Per-sample detail for the strongest (full) edit.
    full = {s["id"]: _score(full_humanize(s["text"])) for s in ai_samples}
    full_evaded_ids = [sid for sid, v in full.items() if v < PRODUCT_THRESHOLD]

    # ---- console ----
    print("=" * 72)
    print("VERUM — Engine A adversarial robustness")
    print("=" * 72)
    print(f"AI samples: {n}   product threshold: {PRODUCT_THRESHOLD:.2f}")
    print(f"Baseline mean AI score: {base_mean:.3f}   "
          f"({base_evaded}/{n} already below threshold before any edit)")
    print("-" * 72)
    print(f"{'edit':<20}{'mean before':<13}{'mean after':<12}{'drop':<8}{'evade <thr'}")
    for name, before, after, drop, evaded in rows:
        print(f"{name:<20}{before:<13.3f}{after:<12.3f}{drop:<8.3f}{evaded}/{n}")
    print("=" * 72)

    # ---- write ROBUSTNESS.md ----
    md: List[str] = []
    md.append("# Verum — Engine A Robustness Report\n")
    md.append("> Generated by `python -m eval.robustness`. This measures how "
              "easily the content detector can be **evaded** by lightly editing "
              "an AI draft. Higher evasion is worse — and we report it honestly "
              "rather than hide it.\n")
    md.append("## What this measures\n")
    md.append("Stylometric AI-text detection keys off *surface* tells (specific "
              "phrasings, the em-dash character, rule-of-three lists, filler "
              "words). Those tells are cheap to remove without changing what the "
              "text actually says. We take every AI sample, apply realistic "
              "edits, and re-score it with the exact code the live API uses.\n")
    md.append(f"- AI samples tested: **{n}**\n"
              f"- Product threshold (a text-only submission crosses into "
              f"“Needs human review” here): **{PRODUCT_THRESHOLD:.2f}**\n"
              f"- Baseline mean AI score: **{base_mean:.3f}** "
              f"({base_evaded}/{n} already sit below the threshold before any "
              f"edit — those are the deliberately hard, lightly-humanized "
              f"samples from the dataset).\n")
    md.append("## Effect of each edit\n")
    md.append("| Edit | What it does | Mean score before | Mean score after | Mean drop | AI samples that now evade (< threshold) |")
    md.append("|------|--------------|-------------------|------------------|-----------|------------------------------------------|")
    for (name, before, after, drop, evaded), (_n, desc, _fn) in zip(rows, TRANSFORMS):
        md.append(f"| **{name}** | {desc} | {before:.3f} | {after:.3f} | "
                  f"{drop:.3f} | {evaded}/{n} |")
    md.append("")
    md.append("## The realistic adversary (all edits at once)\n")
    md.append(f"Applying every edit together, **{len(full_evaded_ids)}/{n}** AI "
              f"samples fall below the review line and read as authentic. "
              f"Per-sample scores after the full edit:\n")
    md.append("| id | baseline score | after full humanize | evades? |")
    md.append("|----|----------------|---------------------|---------|")
    for s in ai_samples:
        sid = s["id"]
        ev = "yes" if full[sid] < PRODUCT_THRESHOLD else "no"
        md.append(f"| {sid} | {base_scores[sid]:.3f} | {full[sid]:.3f} | {ev} |")
    md.append("")
    md.append("## Honest reading of this result\n")
    md.append("- **This is the ceiling of stylometric detection, not a bug in "
              "Verum.** Any content-only detector can be defeated by an applicant "
              "who edits their draft. Marketing claims of “99% accuracy” "
              "quietly ignore exactly this.\n"
              "- **Verum is designed around the limit, not in denial of it.** "
              "Every flag routes to a human and nothing is ever auto-rejected, so "
              "a missed edit costs a second look rather than a person's "
              "opportunity.\n"
              "- **Content is one of three engines.** The submission-behavior "
              "engine (timing, mouse entropy, keystroke rhythm, honeypot) and the "
              "cross-document consistency check are untouched by prose edits, so a "
              "serious evader has to beat all three, not just reword a sentence.\n"
              "- **Takeaway for a reviewer:** treat a low content score as "
              "“no surface tells found,” never as “certainly human.”\n")

    out_path = os.path.join(_HERE, "ROBUSTNESS.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    run()
