"""
services/ml_model.py
====================
SERVE the learned classifier (see eval/train_model.py) with ZERO heavy
dependencies.

The model is a logistic regression exported to `model.json` — just a list of
feature names, a scaler (mean/std), a coefficient per feature, and an intercept.
Scoring is therefore a two-line calculation: standardize, dot-product, sigmoid.
We implement it in pure Python (only the stdlib `math`) so the live API needs
neither numpy nor scikit-learn at run time and still fits the 512MB free-tier
host. Training needs numpy; serving does not.

Design choices that matter for a production-ish service
-------------------------------------------------------
* **Fail safe, never crash.** If `model.json` is missing or malformed, the
  loader returns None and Engine A silently falls back to its hand-tuned
  heuristic. A model file problem must never take down the API.
* **Load once.** The file is parsed a single time and cached.
* **Explainable by construction.** `top_contributions()` returns each feature's
  signed contribution to the log-odds, so the recruiter view can say *why* the
  learned model scored the way it did — the same evidence-first principle the
  heuristic follows.
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from app.services.features import FEATURE_NAMES, extract_vector

# model.json ships next to this module (written by eval/train_model.py).
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")


@lru_cache(maxsize=1)
def load_model(path: str = _MODEL_PATH) -> Optional[Dict]:
    """Load and validate model.json once. Return None if unavailable/invalid.

    Returning None (rather than raising) is deliberate: the caller treats a
    missing model as "use the heuristic", so a bad deploy degrades gracefully
    instead of 500-ing every request.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            m = json.load(fh)
        names = m["feature_names"]
        coef = m["coef"]
        std = m["standardization"]["std"]
        mean = m["standardization"]["mean"]
        # Structural sanity: everything must line up, or we don't trust it.
        if not (len(names) == len(coef) == len(std) == len(mean)):
            return None
        # The features this server can produce must match what the model expects,
        # in the same order — otherwise a retrain with different features could
        # silently score the wrong columns.
        if list(names) != list(FEATURE_NAMES):
            return None
        return m
    except Exception:
        return None


def is_available() -> bool:
    """True if a valid learned model is loaded and ready to score."""
    return load_model() is not None


def _sigmoid(z: float) -> float:
    """Logistic sigmoid, overflow-safe for large |z|."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _standardized(vector: Sequence[float], model: Dict) -> List[float]:
    mean = model["standardization"]["mean"]
    std = model["standardization"]["std"]
    return [(vector[i] - mean[i]) / (std[i] if std[i] else 1.0)
            for i in range(len(vector))]


def score_vector(vector: Sequence[float], model: Optional[Dict] = None) -> Optional[float]:
    """Return P(AI-written) for a pre-computed feature vector, or None if no model."""
    model = model or load_model()
    if model is None:
        return None
    z = model["intercept"]
    coef = model["coef"]
    xs = _standardized(vector, model)
    for i in range(len(coef)):
        z += coef[i] * xs[i]
    return _sigmoid(z)


def score_text(text: str) -> Optional[float]:
    """Convenience: extract features from raw text and score. None if no model."""
    model = load_model()
    if model is None:
        return None
    return score_vector(extract_vector(text), model)


def top_contributions(vector: Sequence[float], model: Optional[Dict] = None,
                      k: int = 3) -> List[Tuple[str, float]]:
    """Return the k features that pushed this prediction hardest.

    Each contribution is `coef_i * standardized_feature_i` — its additive effect
    on the log-odds. Positive pushes toward "AI", negative toward "human". This
    is what makes the learned score explainable: we can name the exact signals
    that drove it, just like the heuristic's evidence receipts.
    """
    model = model or load_model()
    if model is None:
        return []
    names = model["feature_names"]
    coef = model["coef"]
    xs = _standardized(vector, model)
    contribs = [(names[i], coef[i] * xs[i]) for i in range(len(coef))]
    contribs.sort(key=lambda t: abs(t[1]), reverse=True)
    return [(name, c) for name, c in contribs[:k] if abs(c) > 1e-9]
