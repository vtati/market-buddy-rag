"""Shared pytest setup: repo root on sys.path, LLM keys scrubbed so
the whole suite runs in the deterministic extractive mode, and the
egress-proxy NO_PROXY sanitized (bracketed IPv6 entries break httpx
in this sandbox; harmless elsewhere)."""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_llm_keys(monkeypatch):
    """The suite must pass with zero API keys (the demo's keyless mode)."""
    for var in ("OPENAI_API_KEY", "NEBIUS_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _isolated_session_corpus(tmp_path, monkeypatch):
    """Session-corpus isolation: every test sees an EMPTY session
    registry (a tmp dir), never whatever a live demo server ingested
    into data/session/. The precache dir stays real (read-only) so the
    Tesla ingest tests run offline through the cache path. Retrieval
    indexes are dropped around each test so active-corpus state can't
    leak between tests."""
    from src import registry, retrieval

    monkeypatch.setattr(registry, "SESSION_DIR", tmp_path / "session")
    registry.reset_cache()
    retrieval.invalidate_indexes()
    yield
    registry.reset_cache()
    retrieval.invalidate_indexes()
