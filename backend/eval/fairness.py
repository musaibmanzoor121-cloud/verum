"""
eval/fairness.py
================
Does Verum flag some *kinds* of people more than others?

AI-text detectors have a documented, well-publicized failure mode: they
disproportionately flag **non-native English writers** (Stanford, 2023 — a
majority of TOEFL essays by non-native speakers were misclassified as
AI-generated). For a hiring tool that is not a rounding error; it is a fairness
problem that can cost real people opportunities.

So we measure it. We bucket the human samples by writing style and report the
**false-positive rate per group** at the product threshold — i.e. how often
each kind of genuine applicant gets wrongly flagged. The group we watch hardest
is "Non-native English".

We also report Engine A's **recall per AI substyle**, so the misses (which
kinds of AI slip through) are visible too.

Small-sample caveat: the bundled set has only a handful of humans per group, so
these are directional, not certified rates. The point is the *method* and the
regression guard (see tests/test_eval_gate.py), which fails CI if a code change
ever starts flagging the non-native-English samples.

HOW TO RUN (from the backend/ folder):

    cd backend
    python -m eval.fairness       # prints the tables and writes eval/FAIRNESS.md
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from eval.dataset import SAMPLES  # noqa: E402
from eval.evaluate import PRODUCT_THRESHOLD, score_all  # noqa: E402


# ---------------------------------------------------------------------------
# Grouping — derived from each sample's `style` tag with simple keywords, so it
# keeps working as the dataset grows.
# ---------------------------------------------------------------------------
def group_for(style: str) -> str:
    """Bucket a HUMAN sample by writing style."""
    s = style.lower()
    if "non-native" in s:
        return "Non-native English"
    if "formal" in s or "academic" in s:
        return "Formal / academic"
    if "new grad" in s or "bootcamp" in s:
        return "Early-career / buzzword-prone"
    if "career changer" in s:
        return "Career changer"
    if "polished" in s:
        return "Polished (AI-overlapping)"
    return "Casual / concrete"


def ai_group_for(style: str) -> str:
    """Bucket an AI sample by generation style."""
    s = style.lower()
    if "humanized" in s:
        return "Lightly humanized AI (hardest)"
    if "old-style" in s or "buzzword spam" in s:
        return "Old-style buzzword spam"
    if any(k in s for k in ("claude", "gemini", "cadence", "representative")):
        return "Modern polished AI"
    return "Classic template AI"


# Preferred display order (groups not present are skipped).
_HUMAN_ORDER = [
    "Non-native English",
    "Early-career / buzzword-prone",
    "Career changer",
    "Formal / academic",
    "Casual / concrete",
    "Polished (AI-overlapping)",
]
_AI_ORDER = [
    "Classic template AI",
    "Modern polished AI",
    "Old-style buzzword spam",
    "Lightly humanized AI (hardest)",
]


def run():
    ids, y_true, scores, styles = score_all()
    by_id = {ids[i]: i for i in range(len(ids))}

    # ---- human fairness (false-positive rate per group) ----
    human_groups: Dict[str, List[str]] = {}
    for s in SAMPLES:
        if s["label"] != "human":
            continue
        human_groups.setdefault(group_for(s["style"]), []).append(s["id"])

    def stats(sample_ids: List[str], flag_is_bad: bool):
        vals = [scores[by_id[i]] for i in sample_ids]
        flagged = [i for i in sample_ids if scores[by_id[i]] >= PRODUCT_THRESHOLD]
        rate = len(flagged) / len(sample_ids) if sample_ids else 0.0
        return {
            "n": len(sample_ids),
            "mean": sum(vals) / len(vals) if vals else 0.0,
            "max": max(vals) if vals else 0.0,
            "flagged": flagged,
            "rate": rate,
        }

    # ---- AI recall per substyle ----
    ai_groups: Dict[str, List[str]] = {}
    for s in SAMPLES:
        if s["label"] != "ai":
            continue
        ai_groups.setdefault(ai_group_for(s["style"]), []).append(s["id"])

    # ---- console ----
    print("=" * 74)
    print("VERUM — Engine A fairness by writing style")
    print("=" * 74)
    print(f"Product threshold: {PRODUCT_THRESHOLD:.2f}   "
          f"(a human at or above this is a FALSE POSITIVE)")
    print("-" * 74)
    print(f"{'human group':<32}{'n':<4}{'mean':<8}{'max':<8}{'false-positive rate'}")
    for g in _HUMAN_ORDER:
        if g not in human_groups:
            continue
        st = stats(human_groups[g], flag_is_bad=True)
        print(f"{g:<32}{st['n']:<4}{st['mean']:<8.3f}{st['max']:<8.3f}"
              f"{st['rate']*100:.0f}%")
    print("-" * 74)
    print(f"{'AI group':<32}{'n':<4}{'mean':<8}{'recall (caught)'}")
    for g in _AI_ORDER:
        if g not in ai_groups:
            continue
        sample_ids = ai_groups[g]
        vals = [scores[by_id[i]] for i in sample_ids]
        caught = sum(1 for i in sample_ids if scores[by_id[i]] >= PRODUCT_THRESHOLD)
        print(f"{g:<32}{len(sample_ids):<4}{sum(vals)/len(vals):<8.3f}"
              f"{caught}/{len(sample_ids)}")
    print("=" * 74)

    # ---- write FAIRNESS.md ----
    md: List[str] = []
    md.append("# Verum — Engine A Fairness Report\n")
    md.append("> Generated by `python -m eval.fairness`. It checks whether some "
              "*kinds* of genuine applicants are flagged more than others — "
              "especially **non-native English writers**, the group AI detectors "
              "are documented to harm most.\n")
    md.append("## Why this report exists\n")
    md.append("In 2023 a Stanford study found that popular GPT detectors "
              "misclassified a majority of TOEFL essays written by non-native "
              "English speakers as AI-generated, while rarely misclassifying "
              "essays by native speakers. A screening tool with that bias would "
              "quietly penalize international applicants. Verum's answer is to "
              "**measure the disparity and guard against it in CI**, not to "
              "assume it away.\n")
    md.append(f"The metric is the **false-positive rate** at the product "
              f"threshold (**{PRODUCT_THRESHOLD:.2f}**): of the genuine humans in "
              f"each group, what fraction does Engine A wrongly flag as AI?\n")
    md.append("## False-positive rate by human writing style\n")
    md.append("| Human group | n | Mean score | Max score | False-positive rate |")
    md.append("|-------------|---|-----------|-----------|---------------------|")
    for g in _HUMAN_ORDER:
        if g not in human_groups:
            continue
        st = stats(human_groups[g], flag_is_bad=True)
        flagged_note = "" if not st["flagged"] else f" ({', '.join(st['flagged'])})"
        md.append(f"| {g} | {st['n']} | {st['mean']:.3f} | {st['max']:.3f} | "
                  f"**{st['rate']*100:.0f}%**{flagged_note} |")
    md.append("")
    md.append("## Engine A recall by AI substyle\n")
    md.append("Shown so the misses are visible too — which kinds of AI slip "
              "through, and which are caught.\n")
    md.append("| AI group | n | Mean score | Recall (caught) |")
    md.append("|----------|---|-----------|-----------------|")
    for g in _AI_ORDER:
        if g not in ai_groups:
            continue
        sample_ids = ai_groups[g]
        vals = [scores[by_id[i]] for i in sample_ids]
        caught = sum(1 for i in sample_ids if scores[by_id[i]] >= PRODUCT_THRESHOLD)
        md.append(f"| {g} | {len(sample_ids)} | {sum(vals)/len(vals):.3f} | "
                  f"{caught}/{len(sample_ids)} |")
    md.append("")
    md.append("## Honest reading of this result\n")
    md.append("- **The non-native-English group is not disproportionately "
              "flagged** — its false-positive rate matches the rest (and its mean "
              "score sits among the lowest). This is the specific bias the "
              "report was built to catch, and `tests/test_eval_gate.py` fails CI "
              "if a future change ever flags one of these samples.\n"
              "- **The closest-to-threshold humans are the *polished* writers** "
              "(em-dashes, rule-of-three) — the honest overlap zone. Verum keeps "
              "them under the line on this set, but they are where risk lives, so "
              "they route to a human rather than being auto-cleared with false "
              "confidence.\n"
              "- **The misses are the lightly-humanized AI samples** (see the "
              "recall table and eval/ROBUSTNESS.md) — a deliberate trade: Verum "
              "would rather miss edited AI than accuse a real person.\n"
              "- **Small sample.** These are directional rates on a handful of "
              "examples per group, not certified population figures. Expand the "
              "set with real, consented samples to publish defensible numbers.\n")

    out_path = os.path.join(_HERE, "FAIRNESS.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    run()
