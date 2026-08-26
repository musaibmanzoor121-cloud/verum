"""
eval/metrics.py
===============
Pure-Python classification metrics. No numpy, no scikit-learn — so this runs
anywhere the app runs (including the 512MB free-tier host) with zero extra
installs.

The vocabulary, in plain English
---------------------------------
We are asking one yes/no question per document: "Is this AI-written?"
The POSITIVE class is "ai". So for each document there are four outcomes:

    TP (true positive)  : it WAS ai, and we flagged it        -> good catch
    TN (true negative)  : it was human, and we let it through  -> correct
    FP (false positive) : it was HUMAN, but we flagged it      -> a real person
                                                                  wrongly accused
    FN (false negative) : it WAS ai, but we missed it          -> a cheat slips by

For a hiring tool, FP is the one that hurts a person. We surface it loudly
(as the "false-positive rate") because a screening tool that wrongly flags
honest applicants is worse than one that misses a few AI drafts.

Metrics we compute
------------------
    accuracy    = (TP + TN) / total                 overall correctness
    precision   = TP / (TP + FP)                     of the ones we flagged, how many were really AI
    recall      = TP / (TP + FN)                     of all real AI, how many we caught (a.k.a. sensitivity, TPR)
    specificity = TN / (TN + FP)                     of all real humans, how many we cleared
    fpr         = FP / (FP + TN)                      false-positive rate = 1 - specificity  (fairness metric)
    f1          = harmonic mean of precision & recall
    ROC-AUC     = threshold-independent ranking quality (see roc_auc below)
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def confusion(y_true: Sequence[str], y_pred: Sequence[str],
              positive: str = "ai") -> Dict[str, int]:
    """Count TP / TN / FP / FN given true labels and predicted labels."""
    tp = tn = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        t_pos = (t == positive)
        p_pos = (p == positive)
        if t_pos and p_pos:
            tp += 1
        elif not t_pos and not p_pos:
            tn += 1
        elif not t_pos and p_pos:
            fp += 1
        else:  # t_pos and not p_pos
            fn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _safe_div(a: float, b: float) -> float:
    """Divide, returning 0.0 instead of blowing up on a zero denominator."""
    return a / b if b else 0.0


def metrics_from_confusion(cm: Dict[str, int]) -> Dict[str, float]:
    """Turn a TP/TN/FP/FN dict into the standard rate metrics."""
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    total = tp + tn + fp + fn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)              # sensitivity / TPR
    specificity = _safe_div(tn, tn + fp)         # TNR
    fpr = _safe_div(fp, fp + tn)                 # 1 - specificity
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, total)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "f1": f1,
    }


def roc_auc(scores: Sequence[float], y_true: Sequence[str],
            positive: str = "ai") -> float:
    """
    Threshold-independent quality: the probability that a randomly chosen AI
    document gets a HIGHER score than a randomly chosen human one.

    1.0  = perfect ranking (every AI doc scores above every human doc)
    0.5  = no better than a coin flip

    Implemented via the Mann-Whitney U relationship (average rank of the
    positive class), which handles ties correctly and needs no libraries.
    """
    labels = [1 if t == positive else 0 for t in y_true]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")  # AUC undefined if only one class is present

    # Rank all scores (average rank for ties), ranks start at 1.
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # +1 because ranks are 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    sum_ranks_pos = sum(r for r, lab in zip(ranks, labels) if lab == 1)
    u_pos = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return u_pos / (n_pos * n_neg)


def sweep_thresholds(scores: Sequence[float], y_true: Sequence[str],
                     positive: str = "ai",
                     steps: int = 101) -> List[Tuple[float, Dict[str, float]]]:
    """
    Evaluate every threshold from 0.0 to 1.0 and return (threshold, metrics).
    Useful for picking an operating point and for plotting a ROC curve later.
    """
    out: List[Tuple[float, Dict[str, float]]] = []
    for s in range(steps):
        thr = s / (steps - 1)
        y_pred = [positive if sc >= thr else "human" for sc in scores]
        cm = confusion(y_true, y_pred, positive=positive)
        m = metrics_from_confusion(cm)
        m["threshold"] = thr
        out.append((thr, m))
    return out


def render_confusion_table(cm: Dict[str, int]) -> str:
    """A tiny ASCII confusion matrix for the console / markdown report."""
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    return (
        "                        PREDICTED\n"
        "                   AI-flagged   Cleared\n"
        f"  ACTUAL  AI          {tp:>4} (TP)   {fn:>4} (FN)\n"
        f"          Human       {fp:>4} (FP)   {tn:>4} (TN)\n"
    )
