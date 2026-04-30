"""Integration tests for Audio Flamingo containers via HTTP API.

Requires the containers to be running:
  - arena-audio-flamingo           on localhost:8601
  - arena-audio-flamingo-captioner on localhost:8607
  - arena-audio-flamingo-think     on localhost:8608

Run with:
  pytest tests/test_audio_flamingo_api.py -v
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

_SERVICES = {
    "base": {
        "url": "http://localhost:8601",
        "display_name": "Audio Flamingo Next",
        "model_id": "nvidia/audio-flamingo-next-hf",
    },
    "captioner": {
        "url": "http://localhost:8607",
        "display_name": "Audio Flamingo Next Captioner",
        "model_id": "nvidia/audio-flamingo-next-captioner-hf",
    },
    "think": {
        "url": "http://localhost:8608",
        "display_name": "Audio Flamingo Next Think",
        "model_id": "nvidia/audio-flamingo-next-think-hf",
    },
}
_QUESTION = "What sound do you hear?"


def _is_up(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=3.0).status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module", params=list(_SERVICES.keys()))
def service(request):
    svc = _SERVICES[request.param]
    if not _is_up(svc["url"]):
        pytest.skip(
            f"audio-flamingo-{request.param} container not running on {svc['url']}"
        )
    return svc


@pytest.fixture(scope="session")
def sample_wav() -> Path:
    path = Path(__file__).parent / "fixtures" / "sample.wav"
    assert path.exists()
    return path


# ---------------------------------------------------------------------------
# Health / Info
# ---------------------------------------------------------------------------

def test_health(service):
    r = httpx.get(f"{service['url']}/health", timeout=5.0)
    assert r.status_code == 200
    assert r.json()["model"] == service["display_name"]


def test_info(service):
    r = httpx.get(f"{service['url']}/info", timeout=5.0)
    assert r.status_code == 200
    body = r.json()
    assert body["model_id"] == service["model_id"]
    assert body["display_name"] == service["display_name"]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def test_infer_returns_answer(service, sample_wav):
    with sample_wav.open("rb") as f:
        r = httpx.post(
            f"{service['url']}/infer",
            files={"audio": ("sample.wav", f, "audio/wav")},
            data={"question": _QUESTION, "max_new_tokens": "128"},
            timeout=180.0,
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["answer"], str)
    assert len(body["answer"]) > 0
    assert body["model_id"] == service["model_id"]
    assert body["latency_ms"] > 0


def test_infer_rejects_unsupported_format(service, tmp_path):
    bad_file = tmp_path / "audio.xyz"
    bad_file.write_bytes(b"fake")
    with bad_file.open("rb") as f:
        r = httpx.post(
            f"{service['url']}/infer",
            files={"audio": ("audio.xyz", f, "application/octet-stream")},
            data={"question": _QUESTION, "max_new_tokens": "64"},
            timeout=30.0,
        )
    assert r.status_code == 422
