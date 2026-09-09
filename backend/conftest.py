"""
conftest.py (backend/)
======================
Makes the backend package layout importable no matter how pytest is invoked
(`pytest`, `python -m pytest`, from the repo root, or from backend/). We add the
backend/ directory to `sys.path` so that both `app` (the FastAPI application)
and `eval` (the honest-evaluation harness) resolve as top-level packages.

Without this, `from app.main import app` and `from eval.evaluate import ...`
only work under *some* invocation styles, which makes CI flaky. Being explicit
here keeps the test suite portable across machines and the CI runner.
"""

import os
import sys

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
