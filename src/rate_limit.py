"""Public-demo abuse limits (in-memory).

The deployed demo is an open site: in LLM mode every chat question
burns the owner's API key, and ingest is the expensive operation (a
full EDGAR fetch + slice + index build). These limits keep one
visitor — or one script — from draining either budget:

* chat: ``CHAT_LIMIT_PER_HOUR`` questions per visitor IP per hour
  (default 30), plus a global circuit breaker of ``CHAT_DAILY_CAP``
  questions per day across all visitors (default 600);
* ingest: ``INGEST_LIMIT_PER_HOUR`` per visitor IP per hour
  (default 6), and the session corpus itself is capped at
  ``SESSION_COMPANY_CAP`` companies (default 40);
* concurrency: at most ``CHAT_MAX_CONCURRENCY`` chat generations in
  flight (default 4) — excess requests get the limit response rather
  than queueing unboundedly.

State is in-memory per process and resets on redeploy — acceptable
at demo scale, and stated in the README. Client identity is the
first hop of ``X-Forwarded-For`` (the site is reached through the
Vercel rewrite proxy), falling back to the socket peer address.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque

HOUR = 3600.0
DAY = 86400.0


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def client_ip(request) -> str:
    """First X-Forwarded-For hop, else the socket peer address."""
    first = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    if first:
        return first
    return request.client.host if getattr(request, "client", None) else "unknown"


class RateLimiter:
    """Sliding-window limits + a chat concurrency guard.

    ``time_fn`` is injectable so tests drive the windows with a fake
    clock — no sleeping, no live load. All public methods return
    ``None`` when the request may proceed, or the friendly limit
    message when it may not.
    """

    def __init__(self, *, chat_per_hour: int = 30, chat_daily_cap: int = 600,
                 ingest_per_hour: int = 6, session_company_cap: int = 40,
                 max_concurrency: int = 4, time_fn=time.monotonic) -> None:
        self.chat_per_hour = chat_per_hour
        self.chat_daily_cap = chat_daily_cap
        self.ingest_per_hour = ingest_per_hour
        self.session_company_cap = session_company_cap
        self.max_concurrency = max_concurrency
        self._now = time_fn
        self._lock = threading.Lock()
        self._chat_by_ip: dict[str, deque] = {}
        self._ingest_by_ip: dict[str, deque] = {}
        self._chat_global: deque = deque()
        self._in_flight = 0

    @classmethod
    def from_env(cls, time_fn=time.monotonic) -> "RateLimiter":
        return cls(
            chat_per_hour=_env_int("CHAT_LIMIT_PER_HOUR", 30),
            chat_daily_cap=_env_int("CHAT_DAILY_CAP", 600),
            ingest_per_hour=_env_int("INGEST_LIMIT_PER_HOUR", 6),
            session_company_cap=_env_int("SESSION_COMPANY_CAP", 40),
            max_concurrency=_env_int("CHAT_MAX_CONCURRENCY", 4),
            time_fn=time_fn,
        )

    @staticmethod
    def _prune(dq: deque, now: float, window: float) -> None:
        while dq and dq[0] <= now - window:
            dq.popleft()

    def check_chat(self, ip: str) -> str | None:
        """Record one chat question from ``ip``; None = allowed."""
        now = self._now()
        with self._lock:
            dq = self._chat_by_ip.setdefault(ip, deque())
            self._prune(dq, now, HOUR)
            if len(dq) >= self.chat_per_hour:
                return (f"This public demo allows {self.chat_per_hour} "
                        "questions per visitor per hour — you've reached "
                        "that limit. Please try again in a little while; "
                        "the filings aren't going anywhere.")
            self._prune(self._chat_global, now, DAY)
            if len(self._chat_global) >= self.chat_daily_cap:
                return ("This public demo has reached its daily question "
                        "limit — please come back tomorrow.")
            dq.append(now)
            self._chat_global.append(now)
            return None

    def check_ingest(self, ip: str, session_company_count: int,
                     already_registered: bool) -> str | None:
        """Record one ingest start from ``ip``; None = allowed. The
        corpus cap only blocks NEW companies — re-adding one that is
        already registered stays possible (it is the refresh path)."""
        now = self._now()
        with self._lock:
            dq = self._ingest_by_ip.setdefault(ip, deque())
            self._prune(dq, now, HOUR)
            if len(dq) >= self.ingest_per_hour:
                return (f"This public demo allows {self.ingest_per_hour} "
                        "company ingests per visitor per hour — you've "
                        "reached that limit. Please try again in a "
                        "little while.")
            if (not already_registered
                    and session_company_count >= self.session_company_cap):
                return (f"The session corpus is full — this public demo "
                        f"holds at most {self.session_company_cap} "
                        "live-ingested companies at a time. Chat can "
                        "still answer about everything already in it.")
            dq.append(now)
            return None

    def try_acquire_chat(self) -> bool:
        """Take a chat-generation slot; False when all are busy."""
        with self._lock:
            if self._in_flight >= self.max_concurrency:
                return False
            self._in_flight += 1
            return True

    def release_chat(self) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

    def concurrency_message(self) -> str:
        return ("The demo is answering several questions at once right "
                "now — please retry in a few seconds.")
