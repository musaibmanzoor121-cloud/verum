"""
config.py
=========
Central place for all tunable settings. Everything is driven by environment
variables so you can change behavior in deployment WITHOUT touching code.

Why a config module?
--------------------
As a beginner you'll be tempted to hard-code numbers (like "3 seconds") all
over the codebase. That becomes painful to change. Instead we read them once
here, give them sensible defaults, and import `settings` everywhere else.

Read an env var with a default:
    os.getenv("NAME", "default")

We convert strings to the right type (bool/float/int) because env vars are
ALWAYS strings.
"""

import os


def _get_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable ('true'/'1'/'yes' => True)."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    # ---- General ----------------------------------------------------------
    APP_NAME: str = "Verum — Applicant Authenticity Engine"
    APP_VERSION: str = "1.0.0"

    # Comma-separated list of allowed frontend origins for CORS.
    # In production, set this to your deployed frontend URL.
    CORS_ORIGINS: list[str] = [
        o.strip()
        for o in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173",
        ).split(",")
        if o.strip()
    ]

    # ---- Content authenticity (Engine A) ----------------------------------
    # Whether to load a real transformer (GPT-2) to compute perplexity.
    # This is HEAVY (downloads ~500MB, needs torch). It is DISABLED by default
    # so the app starts instantly and fits on free-tier hosts (512MB RAM).
    # When disabled we use a fast statistical proxy instead (see ai_analyzer).
    USE_TRANSFORMER_PERPLEXITY: bool = _get_bool("USE_TRANSFORMER_PERPLEXITY", False)

    # Which HF model to use IF the transformer path is enabled.
    PERPLEXITY_MODEL: str = os.getenv("PERPLEXITY_MODEL", "gpt2")

    # ---- Bot / behavior detection (Engine B) ------------------------------
    # Submissions faster than this many seconds are suspicious (humans can't
    # read + fill a real form this fast).
    MIN_SUBMIT_SECONDS: float = float(os.getenv("MIN_SUBMIT_SECONDS", "3.0"))

    # Simple in-memory rate limit: max requests per IP per window.
    RATE_LIMIT_MAX: int = int(os.getenv("RATE_LIMIT_MAX", "30"))
    RATE_LIMIT_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

    # Optional Redis URL. If set, rate limiting could be backed by Redis for
    # multi-instance deploys. If empty (default) we use in-memory limiting,
    # which is perfectly fine for a single-instance demo.
    REDIS_URL: str = os.getenv("REDIS_URL", "")


# A single shared instance imported everywhere: `from app.config import settings`
settings = Settings()
