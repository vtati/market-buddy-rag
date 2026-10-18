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
