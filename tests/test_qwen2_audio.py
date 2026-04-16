from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from models.qwen2_audio import Qwen2AudioModel
from models.base import InferenceResult


def test_display_name():
    m = Qwen2AudioModel()
    assert m.display_name == "Qwen2-Audio"


def test_model_id():
    m = Qwen2AudioModel()
    assert m.model_id == "Qwen/Qwen2-Audio-7B-Instruct"


def test_load_is_idempotent():
    m = Qwen2AudioModel(device="cpu")

    with patch("models.qwen2_audio.Qwen2AudioForConditionalGeneration") as MockModel, \
         patch("models.qwen2_audio.AutoProcessor") as MockProcessor, \
         patch("models.qwen2_audio.torch") as mock_torch:

        mock_torch.cuda.is_available.return_value = False
        mock_torch.float32 = "float32"
        MockModel.from_pretrained.return_value = MagicMock()
        MockProcessor.from_pretrained.return_value = MagicMock()

        m.load()
        m.load()

        assert MockModel.from_pretrained.call_count == 1


def test_load_falls_back_to_cpu_when_cuda_unavailable():
    m = Qwen2AudioModel(device="cuda")

    with patch("models.qwen2_audio.Qwen2AudioForConditionalGeneration") as MockModel, \
         patch("models.qwen2_audio.AutoProcessor") as MockProcessor, \
         patch("models.qwen2_audio.torch") as mock_torch:

        mock_torch.cuda.is_available.return_value = False
        mock_torch.float32 = "float32"
        MockModel.from_pretrained.return_value = MagicMock()
        MockProcessor.from_pretrained.return_value = MagicMock()

        m.load()

        assert m._device == "cpu"


def test_run_inference_raises_on_unsupported_extension(tmp_path):
    audio_file = tmp_path / "audio.xyz"
    audio_file.write_bytes(b"fake")

    m = Qwen2AudioModel()
    m._model = MagicMock()
    m._processor = MagicMock()
    m._device = "cpu"

    with pytest.raises(ValueError, match="Unsupported audio format"):
        m.run_inference(audio_file, "What do you hear?")


def test_run_inference_raises_if_not_loaded(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake")

    m = Qwen2AudioModel()

    with pytest.raises(RuntimeError, match="not loaded"):
        m.run_inference(audio_file, "What do you hear?")


def test_run_inference_returns_inference_result(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake wav")

    mock_input_ids = MagicMock()
    mock_input_ids.size.return_value = 10
    mock_inputs = MagicMock()
    mock_inputs.input_ids = mock_input_ids

    mock_generated = MagicMock()

    mock_processor = MagicMock()
    mock_processor.feature_extractor.sampling_rate = 16000
    mock_processor.apply_chat_template.return_value = "<chat template text>"
    mock_processor.return_value = mock_inputs
    mock_processor.batch_decode.return_value = ["This is the answer"]

    mock_model = MagicMock()
    mock_model.generate.return_value = mock_generated

    m = Qwen2AudioModel()
    m._model = mock_model
    m._processor = mock_processor
    m._device = "cpu"

    with patch("models.qwen2_audio.torch") as mock_torch, \
         patch("models.qwen2_audio.librosa") as mock_librosa:

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=None)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_torch.inference_mode.return_value = mock_ctx
        mock_librosa.load.return_value = (MagicMock(), 16000)

        result = m.run_inference(audio_file, "What do you hear?")

    assert isinstance(result, InferenceResult)
    assert result.answer == "This is the answer"
    assert result.model_id == "Qwen/Qwen2-Audio-7B-Instruct"
    assert result.latency_ms >= 0


def test_run_inference_loads_audio_with_librosa(tmp_path):
    """音声は librosa.load() でロードして processor に渡す"""
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake wav")

    mock_input_ids = MagicMock()
    mock_input_ids.size.return_value = 10
    mock_inputs = MagicMock()
    mock_inputs.input_ids = mock_input_ids

    mock_processor = MagicMock()
    mock_processor.feature_extractor.sampling_rate = 16000
    mock_processor.apply_chat_template.return_value = "<chat>"
    mock_processor.return_value = mock_inputs
    mock_processor.batch_decode.return_value = ["answer"]

    m = Qwen2AudioModel()
    m._model = MagicMock()
    m._model.generate.return_value = MagicMock()
    m._processor = mock_processor
    m._device = "cpu"

    fake_audio_np = MagicMock()

    with patch("models.qwen2_audio.torch") as mock_torch, \
         patch("models.qwen2_audio.librosa") as mock_librosa:

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=None)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_torch.inference_mode.return_value = mock_ctx
        mock_librosa.load.return_value = (fake_audio_np, 16000)

        m.run_inference(audio_file, "What?")

    # librosa.load が正しいサンプリングレートで呼ばれたか確認
    mock_librosa.load.assert_called_once_with(str(audio_file), sr=16000)

    # processor に audios リストが渡されたか確認
    call_kwargs = mock_processor.call_args
    assert call_kwargs is not None
    assert "audios" in call_kwargs.kwargs or (
        call_kwargs.args and len(call_kwargs.args) > 1
    )
