from __future__ import annotations

import logging
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

from models.base import AudioModel, InferenceResult

_SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
_MIMO_REPO_URL = "https://github.com/XiaomiMiMo/MiMo-Audio.git"
_VENDOR_DIR = Path(__file__).parent.parent / "vendor" / "mimo_audio"
_MODEL_ID = "XiaomiMiMo/MiMo-Audio-7B-Instruct"
_TOKENIZER_ID = "XiaomiMiMo/MiMo-Audio-Tokenizer"

_mimo_audio_cls = None
_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)


def _ensure_mimo_audio_src() -> Path:
    if not _VENDOR_DIR.exists():
        logging.info("Cloning MiMo-Audio source to %s ...", _VENDOR_DIR)
        _VENDOR_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", _MIMO_REPO_URL, str(_VENDOR_DIR)],
            check=True,
        )
    return _VENDOR_DIR


def _inject_flash_attn_stub() -> None:
    """Ensure flash_attn.flash_attn_varlen_func is available as a PyTorch SDPA fallback.

    The MiMo audio tokenizer hard-imports flash_attn_varlen_func. flash_attn is not
    installed in this image (no SM 12.0 wheels exist for 2.x), so we inject a stub
    module with a proper __spec__ so that importlib.util.find_spec does not raise.
    If flash_attn happens to be installed, we still replace varlen_func to avoid
    missing-kernel errors on Blackwell (SM 10.0+).
    """
    import importlib.machinery
    import types
    import torch.nn.functional as F

    def _sdpa_varlen(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        dropout_p=0.0, softmax_scale=None,
        causal=False, window_size=(-1, -1), **_,
    ):
        batch = len(cu_seqlens_q) - 1
        outs = []
        for i in range(batch):
            sq, eq = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
            sk, ek = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
            qi = q[sq:eq].transpose(0, 1).unsqueeze(0)
            ki = k[sk:ek].transpose(0, 1).unsqueeze(0)
            vi = v[sk:ek].transpose(0, 1).unsqueeze(0)
            out = F.scaled_dot_product_attention(
                qi, ki, vi, is_causal=causal, scale=softmax_scale
            )
            outs.append(out.squeeze(0).transpose(0, 1))
        return torch.cat(outs, dim=0)

    if "flash_attn" not in sys.modules:
        stub = types.ModuleType("flash_attn")
        stub.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
        stub.flash_attn_varlen_func = _sdpa_varlen
        sys.modules["flash_attn"] = stub
    else:
        sys.modules["flash_attn"].flash_attn_varlen_func = _sdpa_varlen


def _patch_torchaudio_load() -> None:
    """Patch torchaudio.load to fall back to soundfile when torchcodec is missing.

    torchaudio 2.7.0 defaults to torchcodec; if it is not installed the stock
    load() raises ImportError.  soundfile is already a project dependency and
    handles all formats the audio tokenizer needs (wav, flac, mp3 via libsndfile).
    """
    import soundfile as sf
    import torchaudio
    import torch as _torch

    _orig = torchaudio.load

    def _sf_load(uri, *args, **kwargs):
        try:
            return _orig(uri, *args, **kwargs)
        except (ImportError, RuntimeError):
            data, sr = sf.read(str(uri), dtype="float32", always_2d=True)
            return _torch.from_numpy(data.T.copy()), sr

    torchaudio.load = _sf_load


def _patch_attn_implementation() -> None:
    """Force the LLM to use SDPA instead of flash_attention_2."""
    import importlib
    mod = importlib.import_module("src.mimo_audio.modeling_mimo_audio")
    cls = mod.MiMoAudioForCausalLM
    _orig = cls.from_pretrained.__func__

    @classmethod  # type: ignore[misc]
    def _sdpa_from_pretrained(klass, *args, **kwargs):
        kwargs.setdefault("attn_implementation", "sdpa")
        return _orig(klass, *args, **kwargs)

    cls.from_pretrained = _sdpa_from_pretrained


def _load_mimo_audio_cls():
    global _mimo_audio_cls
    if _mimo_audio_cls is not None:
        return _mimo_audio_cls

    mimo_root = _ensure_mimo_audio_src()
    if str(mimo_root) not in sys.path:
        sys.path.insert(0, str(mimo_root))

    # Apply all patches before importing MimoAudio so they take effect at load time
    _inject_flash_attn_stub()
    _patch_torchaudio_load()
    _patch_attn_implementation()

    from src.mimo_audio.mimo_audio import MimoAudio as _Cls  # type: ignore[import]

    _mimo_audio_cls = _Cls
    return _mimo_audio_cls


def _split_thinking(text: str) -> tuple[str | None, str]:
    m = _THINK_RE.match(text)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    return None, text


class _MimoAudio7BBase(AudioModel):
    def __init__(self, device: str = "cuda") -> None:
        self._requested_device = device
        self._model = None

    @property
    def display_name(self) -> str:
        return "MiMo-Audio-7B"

    @property
    def model_id(self) -> str:
        return _MODEL_ID

    def load(self) -> None:
        if self._model is not None:
            return

        device = self._requested_device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        MimoAudioCls = _load_mimo_audio_cls()
        self._model = MimoAudioCls(_MODEL_ID, _TOKENIZER_ID, device=device)

    def _infer(self, audio_path: Path, question: str, thinking: bool) -> str:
        return self._model.audio_understanding_sft(str(audio_path), question, thinking=thinking)


class MimoAudio7BModel(_MimoAudio7BBase):
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
        answer = self._infer(audio_path, question, thinking=False)
        latency_ms = (time.perf_counter() - t0) * 1000

        return InferenceResult(answer=answer, latency_ms=latency_ms, model_id=_MODEL_ID)


class MimoAudio7BThinkingModel(_MimoAudio7BBase):
    @property
    def display_name(self) -> str:
        return "MiMo-Audio-7B-Thinking"

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
        raw = self._infer(audio_path, question, thinking=True)
        latency_ms = (time.perf_counter() - t0) * 1000

        thinking, answer = _split_thinking(raw)
        return InferenceResult(
            answer=answer,
            latency_ms=latency_ms,
            model_id=_MODEL_ID,
            thinking=thinking,
        )
