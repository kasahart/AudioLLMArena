"""Integration tests for MOSS-Audio containers via HTTP API.

Requires the containers to be running:
  - arena-moss-4b          on localhost:8603
  - arena-moss-8b          on localhost:8604
  - arena-moss-8b-thinking on localhost:8606

Run with:
  pytest tests/test_moss_audio_api.py -v
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

_SERVICES = {
    "4b": {
        "url": "http://localhost:8603",
        "display_name": "MOSS-Audio-4B",
        "model_id": "OpenMOSS-Team/MOSS-Audio-4B-Instruct",
    },
    "8b": {
        "url": "http://localhost:8604",
        "display_name": "MOSS-Audio-8B",
        "model_id": "OpenMOSS-Team/MOSS-Audio-8B-Instruct",
    },
    "8b-thinking": {
        "url": "http://localhost:8606",
        "display_name": "MOSS-Audio-8B-Thinking",
        "model_id": "OpenMOSS-Team/MOSS-Audio-8B-Thinking",
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
            f"arena-moss-{request.param} container not running on {svc['url']}"
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
