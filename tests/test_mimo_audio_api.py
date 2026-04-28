"""Integration tests for MiMo-Audio containers via HTTP API.

These tests require the containers to be running:
  - arena-mimo-audio      on localhost:8613
  - arena-mimo-audio-thinking  on localhost:8614

Run with:
  pytest tests/test_mimo_audio_api.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest
import httpx

_BASE_URL = "http://localhost:8613"
_THINKING_URL = "http://localhost:8614"
_QUESTION = "What sound do you hear?"
_MODEL_ID = "XiaomiMiMo/MiMo-Audio-7B-Instruct"


def _is_up(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=3.0).status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def base_url():
    if not _is_up(_BASE_URL):
        pytest.skip("arena-mimo-audio container not running on port 8613")
    return _BASE_URL


@pytest.fixture(scope="module")
def thinking_url():
    if not _is_up(_THINKING_URL):
        pytest.skip("arena-mimo-audio-thinking container not running on port 8614")
    return _THINKING_URL


@pytest.fixture(scope="session")
def sample_wav() -> Path:
    path = Path(__file__).parent / "fixtures" / "sample.wav"
    assert path.exists()
    return path


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_base_health(base_url):
    r = httpx.get(f"{base_url}/health", timeout=5.0)
    assert r.status_code == 200
    assert r.json()["model"] == "MiMo-Audio-7B"


def test_thinking_health(thinking_url):
    r = httpx.get(f"{thinking_url}/health", timeout=5.0)
    assert r.status_code == 200
    assert r.json()["model"] == "MiMo-Audio-7B-Thinking"


def test_base_info(base_url):
    r = httpx.get(f"{base_url}/info", timeout=5.0)
    assert r.status_code == 200
    body = r.json()
    assert body["model_id"] == _MODEL_ID
    assert body["display_name"] == "MiMo-Audio-7B"


def test_thinking_info(thinking_url):
    r = httpx.get(f"{thinking_url}/info", timeout=5.0)
    assert r.status_code == 200
    body = r.json()
    assert body["model_id"] == _MODEL_ID
    assert body["display_name"] == "MiMo-Audio-7B-Thinking"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def test_base_infer_returns_answer(base_url, sample_wav):
    with sample_wav.open("rb") as f:
        r = httpx.post(
            f"{base_url}/infer",
            files={"audio": ("sample.wav", f, "audio/wav")},
            data={"question": _QUESTION, "max_new_tokens": "128"},
            timeout=120.0,
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["answer"], str)
    assert len(body["answer"]) > 0
    assert body["model_id"] == _MODEL_ID
    assert body["latency_ms"] > 0
    assert "thinking" not in body or body["thinking"] is None


def test_thinking_infer_returns_answer(thinking_url, sample_wav):
    with sample_wav.open("rb") as f:
        r = httpx.post(
            f"{thinking_url}/infer",
            files={"audio": ("sample.wav", f, "audio/wav")},
            data={"question": _QUESTION, "max_new_tokens": "256"},
            timeout=180.0,
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["answer"], str)
    assert len(body["answer"]) > 0
    assert body["model_id"] == _MODEL_ID
    assert body["latency_ms"] > 0


def test_base_infer_rejects_unsupported_format(base_url, tmp_path):
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
