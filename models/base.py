from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class InferenceResult:
    answer: str
    latency_ms: float
    model_id: str
    thinking: str | None = None


class AudioModel(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def run_inference(
        self,
        audio_path: Path,
        question: str,
        max_new_tokens: int = 512,
    ) -> InferenceResult: ...

    @property
    @abstractmethod
    def display_name(self) -> str: ...

    @property
    @abstractmethod
    def model_id(self) -> str: ...
