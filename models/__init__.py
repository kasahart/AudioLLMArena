from __future__ import annotations

from models.base import AudioModel, InferenceResult
from models.audio_flamingo import AudioFlamingoModel

REGISTRY: dict[str, type[AudioModel]] = {
    "Audio Flamingo": AudioFlamingoModel,
}


def list_models() -> list[str]:
    return list(REGISTRY.keys())


def get_model(name: str, device: str = "cuda") -> AudioModel:
    if name not in REGISTRY:
        raise KeyError(name)
    return REGISTRY[name](device=device)


__all__ = ["AudioModel", "InferenceResult", "REGISTRY", "list_models", "get_model"]
