from __future__ import annotations

import re
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoProcessor

from models.base import AudioModel, InferenceResult

_SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)


def _split_thinking(text: str) -> tuple[str | None, str]:
    m = _THINK_RE.match(text)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    return None, text


class _AudioFlamingoBase(AudioModel):
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
        _processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

        try:
            _model = AutoModel.from_pretrained(
                self.model_id,
                dtype=dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )
        except ValueError as e:
            if "audioflamingonext" in str(e).lower():
                raise RuntimeError(
                    "The installed transformers build does not support Audio Flamingo Next. "
                    "Run `uv sync --frozen` to get the required build."
                ) from e
            raise

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

        conversation = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "audio", "path": str(audio_path)},
                    ],
                }
            ]
        ]

        batch = self._processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(self._device)

        if "input_features" in batch:
            batch["input_features"] = batch["input_features"].to(self._model.dtype)

        t0 = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.2,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        prompt_length = batch["input_ids"].shape[1]
        answer = self._processor.batch_decode(
            generated_ids[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return InferenceResult(
            answer=answer,
            latency_ms=latency_ms,
            model_id=self.model_id,
        )


class AudioFlamingoModel(_AudioFlamingoBase):
    @property
    def display_name(self) -> str:
        return "Audio Flamingo Next"

    @property
    def model_id(self) -> str:
        return "nvidia/audio-flamingo-next-hf"


class AudioFlamingoNextCaptionerModel(_AudioFlamingoBase):
    @property
    def display_name(self) -> str:
        return "Audio Flamingo Next Captioner"

    @property
    def model_id(self) -> str:
        return "nvidia/audio-flamingo-next-captioner-hf"


class AudioFlamingoNextThinkModel(_AudioFlamingoBase):
    @property
    def display_name(self) -> str:
        return "Audio Flamingo Next Think"

    @property
    def model_id(self) -> str:
        return "nvidia/audio-flamingo-next-think-hf"

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
