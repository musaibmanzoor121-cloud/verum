"""
middleware/bot_detector.py
==========================
Request-level defenses that run BEFORE your endpoint code:

  1. Rate limiting — stop one IP from hammering the endpoint (the classic
     "auto-apply script spamming hundreds of applications" scenario).
  2. Fingerprint capture — stash the client's user-agent and IP on the request
     so the /analyze endpoint can pass them to the behavioral scorer.

Rate-limit storage
-------------------
By default we use a simple IN-MEMORY sliding window (a dict of IP -> recent
timestamps). That's perfect for a single-instance demo. If you later run
multiple instances behind a load balancer, set REDIS_URL and swap in a Redis
counter so the limit is shared — the seam is here and noted below.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Dict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import settings


class InMemoryRateLimiter:
    """Sliding-window limiter: at most MAX requests per WINDOW seconds per IP."""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.time()
        q = self._hits[key]
        # Drop timestamps older than the window.
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.max_requests:
            return False
        q.append(now)
        return True


class BotDetectionMiddleware(BaseHTTPMiddleware):
    """Applies rate limiting to the analyze endpoints and captures fingerprints."""

    def __init__(self, app):
        super().__init__(app)
        self.limiter = InMemoryRateLimiter(
            settings.RATE_LIMIT_MAX, settings.RATE_LIMIT_WINDOW_SECONDS
        )
        # NOTE: to scale horizontally, initialize a Redis client here when
        # settings.REDIS_URL is set and use INCR + EXPIRE instead of the dict.

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"

        # Make fingerprint available to endpoints via request.state.
        request.state.client_ip = client_ip
        request.state.user_agent = request.headers.get("user-agent", "")

        # Only rate-limit the expensive analysis routes.
        if request.url.path.startswith("/api/analyze"):
            if not self.limiter.allow(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": (
                            f"Rate limit exceeded: max {settings.RATE_LIMIT_MAX} "
                            f"requests per {settings.RATE_LIMIT_WINDOW_SECONDS}s. "
                            f"This pattern itself is a strong bot signal."
                        )
                    },
                )

        return await call_next(request)
