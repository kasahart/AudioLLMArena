"""Integration tests for SALMONN-13B container via HTTP API.

Requires the container to be running:
  - arena-salmonn-13b on localhost:8605

Run with:
  pytest tests/test_salmonn_13b_api.py -v
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

_BASE_URL = "http://localhost:8605"
_MODEL_ID = "tsinghua-ee/SALMONN"
_DISPLAY_NAME = "SALMONN-13B"
_QUESTION = "What sound do you hear?"


def _is_up(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=3.0).status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def base_url():
    if not _is_up(_BASE_URL):
        pytest.skip("arena-salmonn-13b container not running on port 8605")
    return _BASE_URL


@pytest.fixture(scope="session")
def sample_wav() -> Path:
    path = Path(__file__).parent / "fixtures" / "sample.wav"
    assert path.exists()
    return path


# ---------------------------------------------------------------------------
# Health / Info
# ---------------------------------------------------------------------------

def test_health(base_url):
    r = httpx.get(f"{base_url}/health", timeout=5.0)
    assert r.status_code == 200
    assert r.json()["model"] == _DISPLAY_NAME


def test_info(base_url):
    r = httpx.get(f"{base_url}/info", timeout=5.0)
    assert r.status_code == 200
    body = r.json()
    assert body["model_id"] == _MODEL_ID
    assert body["display_name"] == _DISPLAY_NAME


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def test_infer_returns_answer(base_url, sample_wav):
    with sample_wav.open("rb") as f:
        r = httpx.post(
            f"{base_url}/infer",
            files={"audio": ("sample.wav", f, "audio/wav")},
            data={"question": _QUESTION, "max_new_tokens": "128"},
            timeout=180.0,
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["answer"], str)
    assert len(body["answer"]) > 0
    assert body["model_id"] == _MODEL_ID
    assert body["latency_ms"] > 0


def test_infer_rejects_unsupported_format(base_url, tmp_path):
    bad_file = tmp_path / "audio.xyz"
    bad_file.write_bytes(b"fake")
    with bad_file.open("rb") as f:
        r = httpx.post(
            f"{base_url}/infer",
            files={"audio": ("audio.xyz", f, "application/octet-stream")},
            data={"question": _QUESTION, "max_new_tokens": "64"},
            timeout=30.0,
        )
    assert r.status_code == 422
