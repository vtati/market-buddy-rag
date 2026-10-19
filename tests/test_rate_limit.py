"""Item C — public-demo abuse limits.

Unit tests for the in-memory limiter with an injected fake clock
(no sleeping, no live HTTP load), plus thin endpoint tests proving
the API answers 429 with the friendly message before any expensive
work (graph run / ingest job) starts.
"""

import pytest
from fastapi.testclient import TestClient

from src import api, registry
from src.rate_limit import RateLimiter, client_ip


class _Clock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _limiter(clock: _Clock, **kw) -> RateLimiter:
    defaults = dict(chat_per_hour=3, chat_daily_cap=5, ingest_per_hour=2,
                    session_company_cap=2, max_concurrency=2)
    defaults.update(kw)
    return RateLimiter(time_fn=clock, **defaults)


def test_chat_per_ip_hourly_limit_and_window_slide():
    clock = _Clock()
    lim = _limiter(clock)
    for _ in range(3):
        assert lim.check_chat("1.2.3.4") is None
    msg = lim.check_chat("1.2.3.4")
    assert msg is not None and "3 questions per visitor per hour" in msg
    # A different visitor is unaffected.
    assert lim.check_chat("5.6.7.8") is None
    # After the window slides, the first visitor may ask again.
    clock.advance(3601)
    assert lim.check_chat("1.2.3.4") is None


def test_chat_global_daily_cap_is_a_circuit_breaker():
    clock = _Clock()
    lim = _limiter(clock)  # daily cap 5
    for i in range(5):
        assert lim.check_chat(f"10.0.0.{i}") is None
    msg = lim.check_chat("10.0.0.99")  # a fresh IP still trips the breaker
    assert msg is not None and "daily question limit" in msg
    clock.advance(86401)
    assert lim.check_chat("10.0.0.99") is None


def test_ingest_per_ip_limit():
    clock = _Clock()
    lim = _limiter(clock)
    assert lim.check_ingest("1.2.3.4", 0, False) is None
    assert lim.check_ingest("1.2.3.4", 0, False) is None
    msg = lim.check_ingest("1.2.3.4", 0, False)
    assert msg is not None and "2 company ingests per visitor per hour" in msg
    clock.advance(3601)
    assert lim.check_ingest("1.2.3.4", 0, False) is None


def test_session_company_cap_blocks_new_but_not_readds():
    clock = _Clock()
    lim = _limiter(clock)  # cap 2
    msg = lim.check_ingest("1.2.3.4", session_company_count=2,
                           already_registered=False)
    assert msg is not None and "session corpus is full" in msg
    # Re-adding an already-registered company (the refresh path) is fine.
    assert lim.check_ingest("1.2.3.4", session_company_count=2,
                            already_registered=True) is None


def test_concurrency_guard_acquire_release():
    clock = _Clock()
    lim = _limiter(clock)  # max 2 in flight
    assert lim.try_acquire_chat() is True
    assert lim.try_acquire_chat() is True
    assert lim.try_acquire_chat() is False
    lim.release_chat()
    assert lim.try_acquire_chat() is True


def test_client_ip_prefers_first_forwarded_hop():
    class _Req:
        headers = {"x-forwarded-for": "203.0.113.7, 10.0.0.1, 10.0.0.2"}
        client = None
    assert client_ip(_Req()) == "203.0.113.7"


def test_client_ip_falls_back_to_socket_peer():
    class _Client:
        host = "192.0.2.55"

    class _Req:
        headers = {}
        client = _Client()
    assert client_ip(_Req()) == "192.0.2.55"


def test_env_configuration(monkeypatch):
    monkeypatch.setenv("CHAT_LIMIT_PER_HOUR", "7")
    monkeypatch.setenv("CHAT_DAILY_CAP", "70")
    monkeypatch.setenv("INGEST_LIMIT_PER_HOUR", "1")
    monkeypatch.setenv("SESSION_COMPANY_CAP", "9")
    monkeypatch.setenv("CHAT_MAX_CONCURRENCY", "2")
    lim = RateLimiter.from_env()
    assert (lim.chat_per_hour, lim.chat_daily_cap, lim.ingest_per_hour,
            lim.session_company_cap, lim.max_concurrency) == (7, 70, 1, 9, 2)


def test_env_defaults(monkeypatch):
    for var in ("CHAT_LIMIT_PER_HOUR", "CHAT_DAILY_CAP",
                "INGEST_LIMIT_PER_HOUR", "SESSION_COMPANY_CAP",
                "CHAT_MAX_CONCURRENCY"):
        monkeypatch.delenv(var, raising=False)
    lim = RateLimiter.from_env()
    assert (lim.chat_per_hour, lim.chat_daily_cap, lim.ingest_per_hour,
            lim.session_company_cap, lim.max_concurrency) == (30, 600, 6, 40, 4)


@pytest.fixture()
def _zero_chat_limit(monkeypatch):
    monkeypatch.setattr(api, "limiter",
                        RateLimiter(chat_per_hour=0, time_fn=_Clock()))


def test_chat_endpoint_429_with_friendly_message(_zero_chat_limit):
    client = TestClient(api.app, raise_server_exceptions=False)
    resp = client.post("/api/chat", json={
        "question": "What was Apple's total net sales in fiscal 2024?"})
    assert resp.status_code == 429
    assert "public demo allows 0 questions per visitor per hour" \
        in resp.json()["detail"]


@pytest.fixture()
def _full_corpus(monkeypatch):
    monkeypatch.setattr(api, "limiter",
                        RateLimiter(session_company_cap=0, time_fn=_Clock()))


def test_ingest_endpoint_429_when_session_corpus_full(_full_corpus):
    assert "KO" not in registry.session_companies()  # a NEW company
    client = TestClient(api.app, raise_server_exceptions=False)
    resp = client.post("/api/ingest", json={"ticker": "KO"})
    assert resp.status_code == 429
    assert "session corpus is full" in resp.json()["detail"]
