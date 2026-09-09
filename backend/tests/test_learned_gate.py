"""
tests/test_learned_gate.py
==========================
QUALITY GATE for the LEARNED model (the scikit-learn-style logistic regression
trained by eval/train_model.py), the counterpart to test_eval_gate.py which
guards the hand-tuned heuristic.

Two things are checked:

  1. The shipped artifact (app/services/model.json) is structurally valid and
     the pure-Python serving path scores AI text above human text. This is what
     the live API loads when USE_LEARNED_MODEL=true, so if it is broken the gate
     goes red before it ships.

  2. The model's HONEST performance — read from eval/cv_scores.json, the
     leave-one-out cross-validated (out-of-fold) predictions — does not regress.
     Floors sit with headroom under the measured numbers (LOO-CV ROC-AUC ~0.84,
     recall ~0.79). We deliberately do NOT gate overall false-positive rate at
     0% here: on this tiny set the learned model flags one buzzword-heavy
     new-grad human at its 0.5 boundary (7% FPR), which is exactly why the
     hand-tuned, 0%-FPR heuristic remains the shipped default and the learned
     model is opt-in. We DO gate the fairness-critical subgroup that AI
     detectors are documented to harm most (non-native English).

These artifacts are produced by `python -m eval.train_model`. They are committed
to the repo (so serving works and this test runs without numpy), and CI also
regenerates them before running the suite, so the gate always reflects the
current code. If they are somehow absent, the tests skip rather than error.

Run with the rest of the suite:  (from backend/)  pytest -q
"""

import json
import os

import pytest

from eval import evaluate as _ev
from eval import metrics as M

# Locate the committed artifacts relative to the eval package.
_EVAL_DIR = os.path.dirname(os.path.abspath(_ev.__file__))
_CV_PATH = os.path.join(_EVAL_DIR, "cv_scores.json")

# Natural decision boundary for a probability output.
_LEARNED_THRESHOLD = 0.5


def _load_cv():
    if not os.path.exists(_CV_PATH):
        pytest.skip("eval/cv_scores.json not found — run `python -m eval.train_model` "
                    "(or let CI generate it) before this gate can check the model.")
    with open(_CV_PATH, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 1. The shipped model artifact + pure-Python serving path
# ---------------------------------------------------------------------------
def test_model_json_is_valid_and_loads():
    """model.json exists, is well-formed, and its feature list matches the
    extractor — ml_model.load_model() enforces all of that and returns None on
    any mismatch, so `is_available()` is a single trustworthy check."""
    from app.services import ml_model

    if not ml_model.is_available():
        pytest.skip("app/services/model.json not found — run "
                    "`python -m eval.train_model` to generate it.")
    model = ml_model.load_model()
    from app.services.features import FEATURE_NAMES
    assert model["feature_names"] == list(FEATURE_NAMES)
    assert len(model["coef"]) == len(FEATURE_NAMES)


def test_served_model_scores_ai_above_human():
    """End-to-end sanity of the exact code the API runs: a blatantly AI cover
    letter must score higher than a plain, concrete human note."""
    from app.services import ml_model

    if not ml_model.is_available():
        pytest.skip("model.json not available")

    ai_text = (
        "Dear Hiring Manager, I am writing to express my enthusiasm for this "
        "role. As a passionate, results-driven professional, I have honed my "
        "ability to deliver scalable, robust, and innovative solutions. I would "
        "be a valuable addition to your esteemed organization."
    )
    human_text = (
        "I fixed the checkout bug that lost us $4k a week. Took three days. I "
        "write tests and I keep good runbooks. Saw the backend opening and "
        "figured I'd reach out."
    )
    ai_score = ml_model.score_text(ai_text)
    human_score = ml_model.score_text(human_text)
    assert ai_score is not None and human_score is not None
    assert ai_score > human_score, (
        f"learned model ranked human ({human_score:.3f}) at/above AI "
        f"({ai_score:.3f})")
    assert ai_score > 0.5, f"blatant AI text only scored {ai_score:.3f}"


def test_top_contributions_are_explainable():
    """The learned score must be able to name WHY — at least one signed
    contribution — so the recruiter view stays evidence-first."""
    from app.services import ml_model
    from app.services.features import extract_vector

    if not ml_model.is_available():
        pytest.skip("model.json not available")

    ai_text = ("Results-driven, detail-oriented engineer leveraging cutting-edge, "
               "best-in-class solutions to spearhead cross-functional synergy.")
    contribs = ml_model.top_contributions(extract_vector(ai_text), k=3)
    assert contribs, "no contributions returned for obviously buzzword-y text"
    assert any(c > 0 for _, c in contribs), "expected a positive (AI-ward) driver"


# ---------------------------------------------------------------------------
# 2. Honest out-of-fold performance (leave-one-out CV) must not regress
# ---------------------------------------------------------------------------
def _cv_scores_and_labels(cv):
    scores = [r["learned_cv"] for r in cv["rows"]]
    labels = [r["label"] for r in cv["rows"]]
    return scores, labels


def test_learned_cv_ranking_quality_does_not_regress():
    """Out-of-fold ROC-AUC (threshold-independent, so not gameable) stays above a
    floor set below the measured ~0.84."""
    cv = _load_cv()
    scores, labels = _cv_scores_and_labels(cv)
    auc = M.roc_auc(scores, labels)
    assert auc >= 0.80, f"learned LOO-CV ROC-AUC regressed to {auc:.3f} (floor 0.80)"


def test_learned_cv_recall_floor():
    """The learned model must still catch a solid majority of AI out-of-fold.
    Floor is below the measured ~0.79."""
    cv = _load_cv()
    scores, labels = _cv_scores_and_labels(cv)
    y_pred = ["ai" if s >= _LEARNED_THRESHOLD else "human" for s in scores]
    recall = M.metrics_from_confusion(M.confusion(labels, y_pred))["recall"]
    assert recall >= 0.60, f"learned LOO-CV recall dropped to {recall:.2f} (floor 0.60)"


def test_learned_cv_non_native_english_never_flagged():
    """Fairness guard for the population AI detectors are documented to harm
    most: no non-native-English human may cross the learned threshold, even
    out-of-fold."""
    cv = _load_cv()
    from eval.fairness import group_for

    flagged = [
        r["id"] for r in cv["rows"]
        if r["label"] == "human"
        and group_for(r["style"]) == "Non-native English"
        and r["learned_cv"] >= _LEARNED_THRESHOLD
    ]
    assert not flagged, f"non-native English humans flagged by learned model: {flagged}"
