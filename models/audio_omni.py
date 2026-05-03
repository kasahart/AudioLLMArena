from __future__ import annotations

import time
from pathlib import Path

from models.base import AudioModel, InferenceResult

_MODEL_ID = "HKUSTAudio/Audio-Omni"
_SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


class AudioOmniModel(AudioModel):
    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._model = None

    @property
    def display_name(self) -> str:
        return "Audio-Omni"

    @property
    def model_id(self) -> str:
        return _MODEL_ID

    def load(self) -> None:
        if self._model is not None:
            return

        from huggingface_hub import snapshot_download
        from audio_omni import AudioOmni

        model_dir = Path(snapshot_download(_MODEL_ID))
        self._model = AudioOmni(
            str(model_dir / "Audio-Omni.json"),
            str(model_dir / "model.ckpt"),
        )

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

        if self._model is None:
            raise RuntimeError("Model is not loaded. Call load() before run_inference().")

        t0 = time.perf_counter()
        answer = self._model.understand(question, audio=str(audio_path))
        latency_ms = (time.perf_counter() - t0) * 1000

        return InferenceResult(
            answer=answer,
            latency_ms=latency_ms,
            model_id=_MODEL_ID,
        )
