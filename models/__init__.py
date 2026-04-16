from __future__ import annotations

from models.base import AudioModel, InferenceResult
from models.audio_flamingo import AudioFlamingoModel
from models.gemma4_e4b import Gemma4E4BModel
from models.qwen2_audio import Qwen2AudioModel

REGISTRY: dict[str, type[AudioModel]] = {
    "Audio Flamingo": AudioFlamingoModel,
    "Gemma-4-E4B": Gemma4E4BModel,
    "Qwen2-Audio": Qwen2AudioModel,
}


def list_models() -> list[str]:
    return list(REGISTRY.keys())


def get_model(name: str, device: str = "cuda") -> AudioModel:
    if name not in REGISTRY:
        available = ", ".join(REGISTRY.keys())
        raise KeyError(f"Model '{name}' not found. Available: {available}")
    return REGISTRY[name](device=device)


__all__ = ["AudioModel", "InferenceResult", "REGISTRY", "list_models", "get_model"]
