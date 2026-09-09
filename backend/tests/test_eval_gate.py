"""
tests/test_eval_gate.py
=======================
A QUALITY GATE, not a normal unit test.

It runs Engine A over the labeled evaluation set (the exact code path the live
API uses) and fails the build if detection quality regresses — or, most
importantly, if the tool ever starts falsely flagging real people.

The thresholds below are *floors with headroom* under the currently-measured
numbers (ROC-AUC ~0.90, 0% false-positive rate, recall ~0.79). If a future code
change pushes any of them past the line, CI goes red and we find out before it
ships, instead of shipping a detector that quietly got worse or unfair.

Run with the rest of the suite:  (from backend/)  pytest -q
"""

from eval import metrics as M
from eval.evaluate import PRODUCT_THRESHOLD, score_all
from eval.fairness import group_for

# Score every sample once for the whole module (fast: pure regex + stats, no
# network, no model download).
_IDS, _Y_TRUE, _SCORES, _STYLES = score_all()


def _predict_at(threshold):
    return ["ai" if s >= threshold else "human" for s in _SCORES]


def test_ranking_quality_does_not_regress():
    """ROC-AUC is threshold-independent, so it cannot be gamed by moving the
    cutoff. Floor is well below the measured ~0.90."""
    auc = M.roc_auc(_SCORES, _Y_TRUE)
    assert auc >= 0.85, f"ROC-AUC regressed to {auc:.3f} (floor is 0.85)"


def test_zero_false_positives_at_product_threshold():
    """The core fairness promise: at the threshold the product actually uses, no
    genuine human is flagged as AI."""
    y_pred = _predict_at(PRODUCT_THRESHOLD)
    flagged_humans = [
        _IDS[i] for i in range(len(_IDS))
        if _Y_TRUE[i] == "human" and y_pred[i] == "ai"
    ]
    assert not flagged_humans, f"Humans wrongly flagged (FP): {flagged_humans}"


def test_recall_floor_still_catches_ai():
    """We accept lower recall to protect humans, but it must not collapse — a
    detector that flags nothing is useless. Floor is below the measured ~0.79."""
    cm = M.confusion(_Y_TRUE, _predict_at(PRODUCT_THRESHOLD))
    recall = M.metrics_from_confusion(cm)["recall"]
    assert recall >= 0.60, f"Recall dropped to {recall:.2f} (floor is 0.60)"


def test_non_native_english_is_never_flagged():
    """Fairness regression guard for the population AI detectors are documented
    to harm most. If a change ever flags a non-native-English human sample, this
    fails loudly."""
    flagged = [
        _IDS[i] for i in range(len(_IDS))
        if _Y_TRUE[i] == "human"
        and group_for(_STYLES[i]) == "Non-native English"
        and _SCORES[i] >= PRODUCT_THRESHOLD
    ]
    assert not flagged, f"Non-native English samples were flagged: {flagged}"
