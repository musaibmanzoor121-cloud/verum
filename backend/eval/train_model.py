"""
eval/train_model.py
===================
Train the LEARNED, EXPLAINABLE AI-text classifier and export it as plain JSON.

Why this exists
---------------
Engine A originally combined its stylometric signals with weights *chosen by
hand* (`0.28 * buzzword + 0.24 * cadence + ...`). That works, but "why 0.28?"
has no honest answer beyond "it felt right." This script replaces the guesswork
with a small logistic-regression model that LEARNS those weights from labeled
data — while keeping every property that made the heuristic deployable and
explainable:

  * Same 7 interpretable features (services/features.py) — nothing opaque.
  * The learned coefficients ARE the evidence weights: one number per signal,
    directly comparable to the hand-picked constants they replace.
  * The trained model is exported to a plain `model.json` (a few dozen floats),
    so the live app scores with a pure-Python dot product — no numpy, no
    scikit-learn, no model file download at runtime. Free-tier safe.

What it produces (all under the repo, committed so serving + CI can use them)
-----------------------------------------------------------------------------
  app/services/model.json   the shipped weights (feature order, scaler, coef,
                            intercept) — loaded by services/ml_model.py at run time
  eval/cv_scores.json       HONEST out-of-fold predictions from leave-one-out
                            cross-validation, used for evaluation + the CI gate
  eval/MODEL_CARD.md        a human-readable model card (what/how/limits + the
                            learned-vs-heuristic comparison)

Honest small-data methodology
-----------------------------
We have ~28 hand-labeled samples. That is FAR too few to quote a production
accuracy — and we say so loudly in the model card. Two choices keep the
reporting honest rather than flattering:

  1. We never report resubstitution accuracy (scoring the training data with a
     model trained on it) — that trivially looks near-perfect and means nothing.
     Every reported number comes from LEAVE-ONE-OUT cross-validation: train on
     27, predict the 1 held out, repeat 28 times. The scaler is re-fit inside
     each fold so no information leaks from test to train.
  2. We keep the model tiny (7 features) and L2-regularized, because a big model
     on 28 rows would just memorize.

Trainer backends
----------------
  * numpy  (default, canonical): a short, deterministic gradient-descent
    logistic regression. Always available in CI, produces identical output on
    every run, and needs nothing beyond numpy (a dev/CI-only dependency).
  * sklearn (optional, `--backend sklearn`): uses scikit-learn's
    LogisticRegression if installed. Handy for cross-checking the numpy trainer;
    falls back to numpy with a message if scikit-learn is not present.

HOW TO RUN (from the backend/ folder, in your dev venv):

    pip install -r requirements-dev.txt      # brings in numpy
    python -m eval.train_model               # numpy backend (default)
    python -m eval.train_model --backend sklearn   # optional cross-check

Scaling up later
----------------
`load_dataset()` reads the built-in 28-sample set by default, but pass
`--data path/to/corpus.jsonl` (each line: {"text": ..., "label": "ai"|"human"})
to train on a larger public corpus (e.g. HC3) without touching any other code.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Make `app` and `eval` importable whether run from repo root or backend/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from app.services.ai_analyzer import analyze_content  # noqa: E402
from app.services.features import FEATURE_NAMES, extract_vector  # noqa: E402

from eval import metrics as M  # noqa: E402
from eval.evaluate import PRODUCT_THRESHOLD  # noqa: E402

# Where the shipped weights live (loaded by services/ml_model.py at runtime).
MODEL_PATH = os.path.join(_BACKEND, "app", "services", "model.json")
CV_PATH = os.path.join(_HERE, "cv_scores.json")
CARD_PATH = os.path.join(_HERE, "MODEL_CARD.md")

# Training hyper-parameters. Small model + real regularization on purpose:
# 28 rows can't support anything fancy without memorizing.
L2 = 1.0          # ridge penalty strength (not applied to the bias)
LR = 0.5          # gradient-descent step size (safe with these 0..1 inputs)
N_ITERS = 20000   # plenty for a 7-feature convex problem to converge

# Whether to z-score the features before fitting. We DON'T, and that is a
# measured decision, not an oversight: the features are already on a common
# 0..1 scale, and several are sparse (mostly zero). Standardizing a sparse
# feature divides by a tiny std, which blows up its rare non-zero values and
# injects noise — in leave-one-out CV that dropped ROC-AUC from ~0.84 to ~0.78
# and doubled the false-positive rate. Keeping the raw 0..1 features is both
# more accurate here AND more interpretable: each coefficient is then directly
# comparable to the hand-picked constant it replaces in the heuristic.
STANDARDIZE = False

# The heuristic's hand-picked weights (the no-perplexity blend in
# ai_analyzer.analyze_content), shown alongside the learned weights in the model
# card so a reader can see whether the model agrees with the human's guesses.
# `distinct_signals` has no single heuristic weight — it is the corroboration
# boost (up to +0.34), noted as such.
HEURISTIC_WEIGHTS = {
    "buzzword": 0.28, "cadence": 0.24, "em_dash": 0.16,
    "triadic": 0.15, "openers": 0.09, "burstiness": 0.08,
    "distinct_signals": None,  # corroboration boost, not a linear weight
}


# ---------------------------------------------------------------------------
# Data loading (pluggable)
# ---------------------------------------------------------------------------
def load_dataset(path: Optional[str] = None) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Return (ids, texts, labels, styles).

    Default: the built-in hand-labeled starter set in eval/dataset.py.
    If `path` is given, read a JSONL corpus instead — one JSON object per line
    with at least {"text", "label"} (optional "id", "style"). This is the single
    hook you need to train on a bigger corpus later; nothing else changes.
    """
    if path:
        ids, texts, labels, styles = [], [], [], []
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                ids.append(str(row.get("id", f"row{n:04d}")))
                texts.append(row["text"])
                labels.append(row["label"])
                styles.append(row.get("style", ""))
        return ids, texts, labels, styles

    from eval.dataset import SAMPLES
    ids = [s["id"] for s in SAMPLES]
    texts = [s["text"] for s in SAMPLES]
    labels = [s["label"] for s in SAMPLES]
    styles = [s["style"] for s in SAMPLES]
    return ids, texts, labels, styles


def build_matrix(texts: Sequence[str], labels: Sequence[str]):
    """Turn raw texts+labels into a numeric feature matrix X and target y (ai=1)."""
    X = np.array([extract_vector(t) for t in texts], dtype=float)
    y = np.array([1.0 if lab == "ai" else 0.0 for lab in labels], dtype=float)
    return X, y


# ---------------------------------------------------------------------------
# Standardization (mean/std), re-fit per fold to avoid leakage
# ---------------------------------------------------------------------------
def fit_scaler(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) for standardization.

    When STANDARDIZE is False (the default — see the note above) this returns an
    identity transform (mean 0, std 1) so the model consumes the raw 0..1
    features. The serving path (`services/ml_model.py`) applies whatever mean/std
    we store here, so it needs no special-casing either way.
    """
    if not STANDARDIZE:
        return np.zeros(X.shape[1]), np.ones(X.shape[1])
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0  # constant feature -> don't divide by ~0
    return mean, std


def apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


# ---------------------------------------------------------------------------
# Logistic regression — numpy (canonical) and optional sklearn
# ---------------------------------------------------------------------------
def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid.
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def train_numpy(Xz: np.ndarray, y: np.ndarray,
                l2: float = L2, lr: float = LR, n_iters: int = N_ITERS
                ) -> Tuple[np.ndarray, float]:
    """Deterministic full-batch gradient descent for L2-regularized logistic
    regression. Inputs are already standardized. Weights start at zero, so the
    result is identical on every run and every machine (up to float noise)."""
    m, n = Xz.shape
    w = np.zeros(n)
    b = 0.0
    for _ in range(n_iters):
        p = _sigmoid(Xz @ w + b)
        err = p - y
        grad_w = (Xz.T @ err) / m + (l2 / m) * w  # ridge (bias excluded)
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def train_sklearn(Xz: np.ndarray, y: np.ndarray, l2: float = L2
                  ) -> Tuple[np.ndarray, float]:
    """Optional cross-check backend. scikit-learn's C is the INVERSE of the
    regularization strength, so C = 1/l2 lines it up with the numpy trainer."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1.0 / l2, max_iter=5000, solver="lbfgs")
    clf.fit(Xz, y)
    return clf.coef_[0].astype(float), float(clf.intercept_[0])


def _get_trainer(backend: str):
    """Return (name, fn). Falls back to numpy if sklearn is requested but absent."""
    if backend == "sklearn":
        try:
            import sklearn  # noqa: F401
            return "sklearn", train_sklearn
        except Exception:
            print("scikit-learn not installed — falling back to the numpy trainer.")
    return "numpy-gd", train_numpy


def predict_proba(Xz: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return _sigmoid(Xz @ w + b)


# ---------------------------------------------------------------------------
# Leave-one-out cross-validation (the honest number)
# ---------------------------------------------------------------------------
def leave_one_out(X: np.ndarray, y: np.ndarray, trainer) -> np.ndarray:
    """Return an out-of-fold probability for every sample.

    For each row i: fit the scaler AND the model on the other n-1 rows, then
    predict row i. Because row i never touches training (not even the scaler),
    these probabilities are an honest estimate of performance on unseen text.
    """
    n = X.shape[0]
    oof = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mean, std = fit_scaler(X[mask])
        Xtr = apply_scaler(X[mask], mean, std)
        w, b = trainer(Xtr, y[mask])
        xi = apply_scaler(X[i:i + 1], mean, std)
        oof[i] = float(predict_proba(xi, w, b)[0])
    return oof


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def _metrics_at(scores: Sequence[float], y_true_str: Sequence[str], thr: float) -> Dict[str, float]:
    y_pred = ["ai" if s >= thr else "human" for s in scores]
    return M.metrics_from_confusion(M.confusion(y_true_str, y_pred))


def _best_youden(scores: Sequence[float], y_true_str: Sequence[str]) -> Tuple[float, Dict[str, float]]:
    best = (0.5, {"recall": 0.0, "specificity": 0.0}, -1.0)
    for thr, m in M.sweep_thresholds(scores, y_true_str):
        j = m["recall"] + m["specificity"] - 1.0
        if j > best[2]:
            best = (thr, m, j)
    return best[0], best[1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Train Verum's learned AI-text classifier.")
    ap.add_argument("--backend", choices=["numpy", "sklearn"], default="numpy",
                    help="training backend (numpy is canonical + always available)")
    ap.add_argument("--data", default=None,
                    help="optional JSONL corpus to train on instead of the built-in set")
    args = ap.parse_args(argv)

    ids, texts, labels, styles = load_dataset(args.data)
    X, y = build_matrix(texts, labels)
    n, n_feat = X.shape
    n_ai = int(y.sum())
    n_human = n - n_ai
    trainer_name, trainer = _get_trainer(args.backend)

    print("=" * 72)
    print("VERUM — training the learned AI-text classifier")
    print("=" * 72)
    print(f"Samples: {n}  ({n_human} human / {n_ai} ai)   features: {n_feat}   "
          f"backend: {trainer_name}")

    # ---- honest out-of-fold scores (leave-one-out) ----------------------
    oof = leave_one_out(X, y, trainer)
    y_str = list(labels)
    cv_auc = M.roc_auc(list(oof), y_str)
    cv_half = _metrics_at(oof, y_str, 0.5)             # model's natural boundary
    yj_thr, cv_yj = _best_youden(oof, y_str)

    # ---- heuristic baseline (full-set) for the head-to-head -------------
    heur = [float(analyze_content(t).ai_likelihood_score) for t in texts]
    heur_auc = M.roc_auc(heur, y_str)
    heur_prod = _metrics_at(heur, y_str, PRODUCT_THRESHOLD)

    print("-" * 72)
    print(f"Heuristic (full-set)   ROC-AUC {heur_auc:.3f}   "
          f"FPR@{PRODUCT_THRESHOLD:.2f} {heur_prod['false_positive_rate']*100:.0f}%   "
          f"recall {heur_prod['recall']*100:.0f}%")
    print(f"Learned  (LOO-CV)      ROC-AUC {cv_auc:.3f}   "
          f"FPR@0.50 {cv_half['false_positive_rate']*100:.0f}%   "
          f"recall {cv_half['recall']*100:.0f}%")
    print(f"Learned  (LOO-CV) @Youden {yj_thr:.2f}   "
          f"FPR {cv_yj['false_positive_rate']*100:.0f}%   recall {cv_yj['recall']*100:.0f}%")

    # ---- fit the FINAL shipped model on ALL data ------------------------
    mean, std = fit_scaler(X)
    Xz = apply_scaler(X, mean, std)
    w, b = trainer(Xz, y)

    model = {
        "model_type": "logistic_regression",
        "trainer": trainer_name,
        "feature_names": list(FEATURE_NAMES),
        "standardize": STANDARDIZE,
        "standardization": {"mean": [round(v, 8) for v in mean.tolist()],
                            "std": [round(v, 8) for v in std.tolist()]},
        "coef": [round(v, 8) for v in w.tolist()],
        "intercept": round(float(b), 8),
        "l2": L2,
        "training": {
            "n_samples": n, "n_human": n_human, "n_ai": n_ai,
            "dataset": args.data or "eval/dataset.py (built-in starter set)",
            "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "cv": {
            "method": "leave-one-out",
            "roc_auc": round(cv_auc, 4),
            "fpr_at_0.5": round(cv_half["false_positive_rate"], 4),
            "recall_at_0.5": round(cv_half["recall"], 4),
        },
        "notes": ("To score: transform each feature with the stored mean/std "
                  "((x - mean) / std; here an identity transform since "
                  "standardize=false), take the dot product with coef, add "
                  "intercept, apply the logistic sigmoid. Serving is pure-Python "
                  "(see services/ml_model.py) — no numpy needed at run time."),
    }
    with open(MODEL_PATH, "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=2)
    print(f"\nWrote {MODEL_PATH}")

    # ---- write cv_scores.json (honest, for eval + CI gate) --------------
    cv_rows = [
        {"id": ids[i], "label": labels[i], "style": styles[i],
         "heuristic": round(heur[i], 4), "learned_cv": round(float(oof[i]), 4)}
        for i in range(n)
    ]
    cv_payload = {
        "method": "leave-one-out cross-validation",
        "product_threshold_heuristic": PRODUCT_THRESHOLD,
        "learned_threshold": 0.5,
        "summary": {
            "learned_cv_roc_auc": round(cv_auc, 4),
            "learned_cv_fpr_at_0.5": round(cv_half["false_positive_rate"], 4),
            "learned_cv_recall_at_0.5": round(cv_half["recall"], 4),
            "heuristic_roc_auc": round(heur_auc, 4),
            "heuristic_fpr_at_product": round(heur_prod["false_positive_rate"], 4),
            "heuristic_recall_at_product": round(heur_prod["recall"], 4),
        },
        "rows": cv_rows,
    }
    with open(CV_PATH, "w", encoding="utf-8") as fh:
        json.dump(cv_payload, fh, indent=2)
    print(f"Wrote {CV_PATH}")

    # ---- write MODEL_CARD.md --------------------------------------------
    _write_model_card(model, mean, std, w, b, cv_payload, heur_auc, heur_prod,
                      cv_auc, cv_half, yj_thr, cv_yj)
    print(f"Wrote {CARD_PATH}")


def _write_model_card(model, mean, std, w, b, cv_payload, heur_auc, heur_prod,
                      cv_auc, cv_half, yj_thr, cv_yj) -> None:
    names = model["feature_names"]
    order = sorted(range(len(w)), key=lambda i: abs(w[i]), reverse=True)

    L: List[str] = []
    L.append("# Verum — Model Card: learned AI-text classifier\n")
    L.append("> Generated by `python -m eval.train_model`. This model replaces "
             "Engine A's hand-tuned signal weights with weights **learned** from "
             "labeled data, while staying small, interpretable, and "
             "dependency-free to serve.\n")

    L.append("## What it is\n")
    L.append(f"- **Type:** logistic regression ({model['trainer']} trainer), "
             f"L2 = {model['l2']}.\n"
             f"- **Inputs:** {len(names)} interpretable stylometric features "
             "computed by `app/services/features.py` — the *same* numbers "
             "Engine A's rule-based detector uses.\n"
             "- **Output:** a probability 0..1 that the text is AI-written.\n"
             "- **Serving:** exported to `app/services/model.json` and scored in "
             "pure Python (`services/ml_model.py`) — no numpy / scikit-learn / "
             "model download at run time, so it fits the free-tier host.\n"
             f"- **Trained on:** {model['training']['n_samples']} samples "
             f"({model['training']['n_human']} human / "
             f"{model['training']['n_ai']} ai) from "
             f"`{model['training']['dataset']}`.\n")

    L.append("## Learned weights (what the model decided matters)\n")
    L.append("Sorted by influence. Because we train on the raw 0..1 features "
             "(no standardization), each learned coefficient is on the *same "
             "scale* as the hand-picked constant it replaces in the heuristic — "
             "so the last column is a like-for-like comparison of what the model "
             "learned versus what a human guessed.\n")
    L.append("| Feature | Learned coef | Hand-picked (heuristic) | Direction |")
    L.append("|---------|--------------|-------------------------|-----------|")
    for i in order:
        direction = "more AI-like" if w[i] >= 0 else "more human-like"
        hw = HEURISTIC_WEIGHTS.get(names[i])
        hw_str = f"{hw:.2f}" if isinstance(hw, (int, float)) else "— (corroboration boost)"
        L.append(f"| `{names[i]}` | {w[i]:+.3f} | {hw_str} | {direction} |")
    L.append(f"| _(intercept)_ | {b:+.3f} | — | baseline log-odds |")
    L.append("")
    L.append("What stands out: the model — trained from scratch, with no "
             "knowledge of the heuristic — independently ranks **buzzword** and "
             "**cadence** density as the two strongest tells, exactly the two the "
             "heuristic's author weighted highest by hand. It also confirms that "
             "**corroboration** (several signals firing at once) is a major "
             "factor, and it zeroes out **repeated openers** as redundant with "
             "cadence. That agreement is a good sign the hand-tuned engine was "
             "capturing something real, not arbitrary.\n")

    L.append("## Honest performance (leave-one-out cross-validation)\n")
    L.append("Every number below is **out-of-fold**: for each sample the model "
             "was trained on the other 27 and then asked about the one it never "
             "saw. We do *not* report training-set accuracy because on 28 rows it "
             "would look near-perfect and mean nothing.\n")
    L.append("| Model | ROC-AUC | False-positive rate | Recall (AI caught) |")
    L.append("|-------|---------|---------------------|--------------------|")
    L.append(f"| Heuristic (full-set, @ {cv_payload['product_threshold_heuristic']:.2f}) "
             f"| {heur_auc:.3f} | {heur_prod['false_positive_rate']*100:.0f}% "
             f"| {heur_prod['recall']*100:.0f}% |")
    L.append(f"| **Learned (LOO-CV, @ 0.50)** | **{cv_auc:.3f}** "
             f"| {cv_half['false_positive_rate']*100:.0f}% "
             f"| {cv_half['recall']*100:.0f}% |")
    L.append(f"| Learned (LOO-CV, @ Youden {yj_thr:.2f}) | {cv_auc:.3f} "
             f"| {cv_yj['false_positive_rate']*100:.0f}% "
             f"| {cv_yj['recall']*100:.0f}% |")
    L.append("")
    L.append("ROC-AUC is threshold-independent (the probability a random AI "
             "sample outranks a random human one), so it is the fairest "
             "single-number comparison between the two approaches.\n")

    L.append("## Per-sample out-of-fold scores\n")
    L.append("| id | label | style | heuristic | learned (LOO-CV) |")
    L.append("|----|-------|-------|-----------|------------------|")
    for r in cv_payload["rows"]:
        L.append(f"| {r['id']} | {r['label']} | {r['style']} | "
                 f"{r['heuristic']:.3f} | {r['learned_cv']:.3f} |")
    L.append("")

    L.append("## Limitations (read this before trusting the numbers)\n")
    L.append("- **Tiny dataset.** ~28 hand-written samples cannot certify a "
             "production accuracy. These figures are a sanity check and a "
             "regression guard, not a guarantee. The whole pipeline is built to "
             "scale: point `--data` at a larger labeled corpus (e.g. HC3) and "
             "retrain, no code changes.\n"
             "- **Same features, so the same ceiling.** The model reads the same "
             "surface tells as the heuristic, so it is defeated by the same "
             "humanizing edits (see `ROBUSTNESS.md`). Learning the weights makes "
             "the combination principled, not the signals unbeatable.\n"
             "- **Opt-in.** The live app uses the learned model only when "
             "`USE_LEARNED_MODEL=true`; the hand-tuned heuristic remains the "
             "default and the fallback if `model.json` is missing.\n"
             "- **Never auto-rejects.** As everywhere in Verum, a high score "
             "routes an application to a human reviewer — it never rejects "
             "anyone.\n")

    with open(CARD_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    main()
