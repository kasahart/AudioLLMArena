from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from models.base import InferenceResult
from models.mimo_audio import (
    MimoAudio7BModel,
    MimoAudio7BThinkingModel,
    _MODEL_ID,
    _split_thinking,
)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def test_base_display_name():
    assert MimoAudio7BModel().display_name == "MiMo-Audio-7B"


def test_base_model_id():
    assert MimoAudio7BModel().model_id == _MODEL_ID


def test_thinking_display_name():
    assert MimoAudio7BThinkingModel().display_name == "MiMo-Audio-7B-Thinking"


def test_thinking_model_id():
    assert MimoAudio7BThinkingModel().model_id == _MODEL_ID


# ---------------------------------------------------------------------------
# _split_thinking
# ---------------------------------------------------------------------------

def test_split_thinking_extracts_think_block():
    text = "<think>some reasoning</think> Final answer."
    thinking, answer = _split_thinking(text)
    assert thinking == "some reasoning"
    assert answer == "Final answer."


def test_split_thinking_returns_none_when_no_block():
    text = "Direct answer."
    thinking, answer = _split_thinking(text)
    assert thinking is None
    assert answer == "Direct answer."


def test_split_thinking_multiline():
    text = "<think>\nline1\nline2\n</think>\nAnswer"
    thinking, answer = _split_thinking(text)
    assert "line1" in thinking
    assert answer == "Answer"


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------

def test_load_is_idempotent():
    m = MimoAudio7BModel(device="cpu")
    mock_mimo_cls = MagicMock()
    mock_instance = MagicMock()
    mock_mimo_cls.return_value = mock_instance

    with patch("models.mimo_audio._load_mimo_audio_cls", return_value=mock_mimo_cls), \
         patch("models.mimo_audio.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        m.load()
        m.load()

    assert mock_mimo_cls.call_count == 1


def test_load_falls_back_to_cpu():
    m = MimoAudio7BModel(device="cuda")
    mock_mimo_cls = MagicMock()

    with patch("models.mimo_audio._load_mimo_audio_cls", return_value=mock_mimo_cls), \
         patch("models.mimo_audio.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        m.load()

    # MimoAudio should have been called with device="cpu"
    _, kwargs = mock_mimo_cls.call_args
    assert kwargs.get("device") == "cpu" or mock_mimo_cls.call_args[0][2] == "cpu"


# ---------------------------------------------------------------------------
# run_inference guards
# ---------------------------------------------------------------------------

def test_run_inference_raises_on_unsupported_extension(tmp_path):
    audio_file = tmp_path / "audio.xyz"
    audio_file.write_bytes(b"fake")

    m = MimoAudio7BModel()
    m._model = MagicMock()

    with pytest.raises(ValueError, match="Unsupported audio format"):
        m.run_inference(audio_file, "What do you hear?")


def test_run_inference_raises_if_not_loaded(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake")

    m = MimoAudio7BModel()
    with pytest.raises(RuntimeError, match="not loaded"):
        m.run_inference(audio_file, "What do you hear?")


def test_thinking_run_inference_raises_on_unsupported_extension(tmp_path):
    audio_file = tmp_path / "audio.xyz"
    audio_file.write_bytes(b"fake")

    m = MimoAudio7BThinkingModel()
    m._model = MagicMock()

    with pytest.raises(ValueError, match="Unsupported audio format"):
        m.run_inference(audio_file, "What do you hear?")


# ---------------------------------------------------------------------------
# run_inference happy path (mocked)
# ---------------------------------------------------------------------------

def _make_model_with_mock(variant, answer_text: str):
    mock_model = MagicMock()
    mock_model.audio_understanding_sft.return_value = answer_text
    m = variant(device="cpu")
    m._model = mock_model
    return m


def test_base_run_inference_returns_result(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake wav")

    m = _make_model_with_mock(MimoAudio7BModel, "I hear music.")
    result = m.run_inference(audio_file, "What do you hear?")

    assert isinstance(result, InferenceResult)
    assert result.answer == "I hear music."
    assert result.model_id == _MODEL_ID
    assert result.latency_ms >= 0
    assert result.thinking is None
    # Must call with thinking=False
    m._model.audio_understanding_sft.assert_called_once_with(
        str(audio_file), "What do you hear?", thinking=False
    )


def test_thinking_run_inference_splits_think_block(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake wav")

    raw = "<think>my reasoning</think> The answer."
    m = _make_model_with_mock(MimoAudio7BThinkingModel, raw)
    result = m.run_inference(audio_file, "What do you hear?")

    assert isinstance(result, InferenceResult)
    assert result.answer == "The answer."
    assert result.thinking == "my reasoning"
    assert result.model_id == _MODEL_ID
    # Must call with thinking=True
    m._model.audio_understanding_sft.assert_called_once_with(
        str(audio_file), "What do you hear?", thinking=True
    )


def test_thinking_run_inference_no_think_block(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake wav")

    m = _make_model_with_mock(MimoAudio7BThinkingModel, "Direct answer.")
    result = m.run_inference(audio_file, "What do you hear?")

    assert result.answer == "Direct answer."
    assert result.thinking is None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_contains_mimo_models():
    from models import list_models
    models = list_models()
    assert "MiMo-Audio-7B" in models
    assert "MiMo-Audio-7B-Thinking" in models


# ---------------------------------------------------------------------------
# GPU end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_api_base_inference(sample_wav: Path) -> None:
    """Integration test: POST to running arena-mimo-audio container (port 8613)."""
    import httpx

    url = "http://localhost:8613"
    try:
        r = httpx.get(f"{url}/health", timeout=5.0)
        assert r.status_code == 200, "Container not healthy"
    except Exception as exc:
        pytest.skip(f"Container not reachable: {exc}")

    with sample_wav.open("rb") as f:
        r = httpx.post(
            f"{url}/infer",
            files={"audio": ("sample.wav", f, "audio/wav")},
            data={"question": "What sound do you hear?", "max_new_tokens": 128},
            timeout=120.0,
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["answer"], str)
    assert len(body["answer"]) > 0
    assert body["model_id"] == _MODEL_ID
    assert body.get("thinking") is None


@pytest.mark.gpu
def test_api_thinking_inference(sample_wav: Path) -> None:
    """Integration test: POST to running arena-mimo-audio-thinking container (port 8614)."""
    import httpx

    url = "http://localhost:8614"
    try:
        r = httpx.get(f"{url}/health", timeout=5.0)
        assert r.status_code == 200, "Container not healthy"
    except Exception as exc:
        pytest.skip(f"Container not reachable: {exc}")

    with sample_wav.open("rb") as f:
        r = httpx.post(
            f"{url}/infer",
            files={"audio": ("sample.wav", f, "audio/wav")},
            data={"question": "What sound do you hear?", "max_new_tokens": 256},
            timeout=180.0,
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["answer"], str)
    assert len(body["answer"]) > 0
    assert body["model_id"] == _MODEL_ID
