from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from transformers import WhisperFeatureExtractor

from models.base import AudioModel, InferenceResult

_MODEL_ID = "tsinghua-ee/SALMONN"
_SALMONN_REPO_URL = "https://github.com/bytedance/SALMONN.git"
_SALMONN_BRANCH = "salmonn"
_VENDOR_DIR = Path(__file__).parent.parent / "vendor" / "salmonn"
_SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

_salmonn_cls = None


def _ensure_salmonn_src() -> Path:
    if not _VENDOR_DIR.exists():
        logging.info("Cloning SALMONN source to %s ...", _VENDOR_DIR)
        _VENDOR_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git", "clone", "-b", _SALMONN_BRANCH, "--depth", "1",
                _SALMONN_REPO_URL, str(_VENDOR_DIR),
            ],
            check=True,
        )
    return _VENDOR_DIR


def _load_salmonn_class():
    """Import the SALMONN class, isolating its models.* namespace from ours."""
    global _salmonn_cls
    if _salmonn_cls is not None:
        return _salmonn_cls

    salmonn_root = _ensure_salmonn_src()

    # Back up our models.* entries so SALMONN can temporarily own the namespace
    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k == "models" or k.startswith("models.")}

    sys.path.insert(0, str(salmonn_root))
    try:
        from models.salmonn import SALMONN  # loads from vendor/salmonn/models/

        # Move SALMONN's models.* under _salmonn_src.* to free the namespace
        for k in [k for k in list(sys.modules) if k == "models" or k.startswith("models.")]:
            sys.modules["_salmonn_src" + k[len("models"):]] = sys.modules.pop(k)

        _salmonn_cls = SALMONN
        return SALMONN

    finally:
        sys.path.remove(str(salmonn_root))
        # Restore our package (only keys that aren't already back)
        for k, v in saved.items():
            sys.modules.setdefault(k, v)


class SALMONNModel(AudioModel):
    def __init__(self, device: str = "cuda") -> None:
        self._requested_device = device
        self._device: str | None = None
        self._model = None
        self._wav_processor = None

    @property
    def display_name(self) -> str:
        return "SALMONN-13B"

    @property
    def model_id(self) -> str:
        return _MODEL_ID

    def load(self) -> None:
        if self._model is not None:
            return

        device = self._requested_device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self._device = device

        beats_path = os.environ.get("SALMONN_BEATS_PATH", "")
        if not beats_path:
            raise RuntimeError(
                "BEATs checkpoint not configured. "
                "Download BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt from the link in "
                "https://github.com/bytedance/SALMONN/tree/salmonn "
                "and set the environment variable SALMONN_BEATS_PATH=/path/to/beats.pt"
            )

        ckpt_path = hf_hub_download(repo_id=_MODEL_ID, filename="salmonn_v1.pth")

        SALMONN = _load_salmonn_class()

        config = {
            "llama_path": "lmsys/vicuna-13b-v1.1",
            "whisper_path": "openai/whisper-large-v2",
            "beats_path": beats_path,
            "ckpt": ckpt_path,
            "lora": True,
            "lora_rank": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "low_resource": device == "cpu",
        }

        _model = SALMONN.from_config(config)
        if device != "cpu":
            _model = _model.to(device)
        _model.eval()

        _wav_processor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")

        self._wav_processor = _wav_processor
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

        if self._model is None or self._wav_processor is None:
            raise RuntimeError("Model is not loaded. Call load() before run_inference().")

        samples = self._prepare_audio(audio_path)

        ctx = (
            torch.cuda.amp.autocast(dtype=torch.float16)
            if self._device != "cpu"
            else contextlib.nullcontext()
        )

        t0 = time.perf_counter()
        with torch.inference_mode(), ctx:
            outputs = self._model.generate(
                samples=samples,
                generate_cfg={
                    "max_new_tokens": max_new_tokens,
                    "num_beams": 4,
                    "do_sample": False,
                    "temperature": 1.0,
                    "top_p": 0.9,
                    "repetition_penalty": 1.0,
                    "length_penalty": 1.0,
                },
                prompts=[question],
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        return InferenceResult(
            answer=outputs[0],
            latency_ms=latency_ms,
            model_id=_MODEL_ID,
        )

    def _prepare_audio(self, audio_path: Path) -> dict:
        import librosa
        target_sr: int = self._wav_processor.sampling_rate  # 16000 Hz (Whisper requirement)
        audio, sr = sf.read(str(audio_path))
        if audio.ndim == 2:
            audio = audio[:, 0]
        if sr != target_sr:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        if len(audio) < target_sr:
            audio = np.concatenate([audio, np.zeros(target_sr - len(audio))])
        audio = audio[: target_sr * 30]

        spectrogram = self._wav_processor(
            audio, sampling_rate=target_sr, return_tensors="pt"
        )["input_features"]

        samples = {
            "spectrogram": spectrogram,
            "raw_wav": torch.from_numpy(audio).float().unsqueeze(0),
            "padding_mask": torch.zeros(len(audio), dtype=torch.bool).unsqueeze(0),
        }

        if self._device != "cpu":
            samples = {k: v.to(self._device) for k, v in samples.items()}

        return samples
