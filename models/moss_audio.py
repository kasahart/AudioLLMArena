from __future__ import annotations

import logging
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from models.base import AudioModel, InferenceResult

_SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
_SAMPLE_RATE = 16000
_MOSS_REPO_URL = "https://github.com/OpenMOSS/MOSS-Audio.git"
_VENDOR_DIR = Path(__file__).parent.parent / "vendor" / "moss_audio"

_moss_audio_cls = None
_moss_processor_cls = None


def _load_audio(path: Path, sample_rate: int = _SAMPLE_RATE) -> np.ndarray:
    import librosa
    return librosa.load(str(path), sr=sample_rate, mono=True)[0]


def _ensure_moss_audio_src() -> Path:
    if not _VENDOR_DIR.exists():
        logging.info("Cloning MOSS-Audio source to %s ...", _VENDOR_DIR)
        _VENDOR_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", _MOSS_REPO_URL, str(_VENDOR_DIR)],
            check=True,
        )
    return _VENDOR_DIR


def _load_moss_classes():
    """Import MossAudioModel and MossAudioProcessor from the cloned GitHub source."""
    global _moss_audio_cls, _moss_processor_cls
    if _moss_audio_cls is not None and _moss_processor_cls is not None:
        return _moss_audio_cls, _moss_processor_cls

    moss_root = _ensure_moss_audio_src()

    if str(moss_root) not in sys.path:
        sys.path.insert(0, str(moss_root))

    from src.modeling_moss_audio import MossAudioModel as _ModelCls  # type: ignore[import]
    from src.processing_moss_audio import MossAudioProcessor as _ProcCls  # type: ignore[import]

    _moss_audio_cls = _ModelCls
    _moss_processor_cls = _ProcCls
    return _moss_audio_cls, _moss_processor_cls


class _MossAudioBase(AudioModel):
    """Shared implementation for all MOSS-Audio variants."""

    def __init__(self, device: str = "cuda") -> None:
        self._requested_device = device
        self._device: str | None = None
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return

        device = self._requested_device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self._device = device

        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        # Both model and processor loaded from the same GitHub source so
        # their token counts stay in sync.
        MossAudioModelCls, MossAudioProcessorCls = _load_moss_classes()

        _processor = MossAudioProcessorCls.from_pretrained(
            self.model_id,
            enable_time_marker=True,
        )

        _model = MossAudioModelCls.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )

        if device != "cuda":
            _model = _model.to(device)

        _model.eval()

        self._processor = _processor
        self._model = _model

    def run_inference(
        self,
        audio_path: Path,
        question: str,
        max_new_tokens: int = 512,
    ) -> InferenceResult:
        if audio_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format: {audio_path.suffix}. "
                f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
            )

        if self._model is None or self._processor is None:
            raise RuntimeError("Model is not loaded. Call load() before run_inference().")

        raw_audio = _load_audio(audio_path, sample_rate=self._processor.config.mel_sr)

        inputs = self._processor(
            text=question,
            audios=[raw_audio],
            return_tensors="pt",
        )
        inputs = inputs.to(self._device)

        if inputs.get("audio_data") is not None:
            inputs["audio_data"] = inputs["audio_data"].to(self._model.dtype)

        inputs["audio_input_mask"] = (
            inputs["input_ids"] == self._processor.audio_token_id
        )

        input_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                top_k=50,
                use_cache=True,
                pad_token_id=self._model.config.eos_token_id,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        answer = self._processor.decode(
            generated_ids[0, input_len:],
            skip_special_tokens=True,
        )

        return InferenceResult(
            answer=answer,
            latency_ms=latency_ms,
            model_id=self.model_id,
        )


_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)


def _split_thinking(text: str) -> tuple[str | None, str]:
    """Split <think>...</think> block from the final answer."""
    m = _THINK_RE.match(text)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    return None, text


class MossAudio4BModel(_MossAudioBase):
    @property
    def display_name(self) -> str:
        return "MOSS-Audio-4B"

    @property
    def model_id(self) -> str:
        return "OpenMOSS-Team/MOSS-Audio-4B-Instruct"


class MossAudio8BModel(_MossAudioBase):
    @property
    def display_name(self) -> str:
        return "MOSS-Audio-8B"

    @property
    def model_id(self) -> str:
        return "OpenMOSS-Team/MOSS-Audio-8B-Instruct"


class MossAudio8BThinkingModel(_MossAudioBase):
    @property
    def display_name(self) -> str:
        return "MOSS-Audio-8B-Thinking"

    @property
    def model_id(self) -> str:
        return "OpenMOSS-Team/MOSS-Audio-8B-Thinking"

    def run_inference(
        self,
        audio_path: Path,
        question: str,
        max_new_tokens: int = 512,
    ) -> InferenceResult:
        result = super().run_inference(audio_path, question, max_new_tokens)
        thinking, answer = _split_thinking(result.answer)
        return InferenceResult(
            answer=answer,
            latency_ms=result.latency_ms,
            model_id=result.model_id,
            thinking=thinking,
        )
