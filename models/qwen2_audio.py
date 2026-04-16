from __future__ import annotations

import time
from pathlib import Path

import librosa
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from models.base import AudioModel, InferenceResult

_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
_SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


class Qwen2AudioModel(AudioModel):
    def __init__(self, device: str = "cuda") -> None:
        self._requested_device = device
        self._device: str | None = None
        self._model = None
        self._processor = None

    @property
    def display_name(self) -> str:
        return "Qwen2-Audio"

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

        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        _processor = AutoProcessor.from_pretrained(_MODEL_ID)
        _model = Qwen2AudioForConditionalGeneration.from_pretrained(
            _MODEL_ID,
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

        audio_np = librosa.load(
            str(audio_path),
            sr=self._processor.feature_extractor.sampling_rate,
        )[0]

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": str(audio_path)},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self._processor(
            text=text,
            audios=[audio_np],
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        prompt_len = inputs.input_ids.size(1)

        t0 = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        answer = self._processor.batch_decode(
            generated_ids[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return InferenceResult(
            answer=answer,
            latency_ms=latency_ms,
            model_id=_MODEL_ID,
        )
