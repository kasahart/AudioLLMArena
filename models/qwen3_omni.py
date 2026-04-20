from __future__ import annotations

import re
import time
from pathlib import Path

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

from models.base import AudioModel, InferenceResult

_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
_CAPTIONER_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Captioner"
_THINKING_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Thinking"

_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)


def _split_thinking(text: str) -> tuple[str | None, str]:
    m = _THINK_RE.match(text)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    return None, text
_SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


class Qwen3OmniModel(AudioModel):
    def __init__(self, device: str = "cuda") -> None:
        self._requested_device = device
        self._device: str | None = None
        self._model = None
        self._processor = None

    @property
    def display_name(self) -> str:
        return "Qwen3-Omni"

    @property
    def model_id(self) -> str:
        return _MODEL_ID

    def _generate(self, inputs, max_new_tokens: int):
        with torch.inference_mode():
            return self._model.generate(
                **inputs,
                return_audio=False,
                thinker_max_new_tokens=max_new_tokens,
                thinker_do_sample=False,
                use_audio_in_video=False,
            )

    def load(self) -> None:
        if self._model is not None:
            return

        device = self._requested_device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self._device = device

        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self._processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_id)
        self._model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            attn_implementation="sdpa",
        )
        if device != "cuda":
            self._model = self._model.to(device)
        self._model.eval()

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
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(audio_path)},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, _, _ = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self._processor(
            text=text,
            audio=audios,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        inputs = inputs.to(self._model.device).to(self._model.dtype)

        prompt_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        output = self._generate(inputs, max_new_tokens)
        latency_ms = (time.perf_counter() - t0) * 1000

        # generate with return_audio=False returns a tensor directly
        token_ids = output[0] if isinstance(output, tuple) else output
        if hasattr(token_ids, "sequences"):
            token_ids = token_ids.sequences

        answer = self._processor.batch_decode(
            token_ids[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return InferenceResult(
            answer=answer,
            latency_ms=latency_ms,
            model_id=self.model_id,
        )


class Qwen3OmniThinkingModel(Qwen3OmniModel):
    @property
    def display_name(self) -> str:
        return "Qwen3-Omni-Thinking"

    @property
    def model_id(self) -> str:
        return _THINKING_MODEL_ID

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


class Qwen3OmniCaptionerModel(Qwen3OmniModel):
    @property
    def display_name(self) -> str:
        return "Qwen3-Omni-Captioner"

    @property
    def model_id(self) -> str:
        return _CAPTIONER_MODEL_ID

    def _generate(self, inputs, max_new_tokens: int):
        with torch.inference_mode():
            return self._model.generate(
                **inputs,
                return_audio=False,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_audio_in_video=False,
            )
