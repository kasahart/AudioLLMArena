from __future__ import annotations

from models.base import AudioModel, InferenceResult

_REGISTRY: dict[str, str] = {
    "Audio Flamingo Next":      "models.audio_flamingo.AudioFlamingoModel",
    "Audio Flamingo Next Captioner": "models.audio_flamingo.AudioFlamingoNextCaptionerModel",
    "Audio Flamingo Next Think":     "models.audio_flamingo.AudioFlamingoNextThinkModel",
    "Gemma-4-E4B":    "models.gemma4_e4b.Gemma4E4BModel",
    "MOSS-Audio-4B":  "models.moss_audio.MossAudio4BModel",
    "MOSS-Audio-8B":          "models.moss_audio.MossAudio8BModel",
    "MOSS-Audio-8B-Thinking": "models.moss_audio.MossAudio8BThinkingModel",
    "Qwen2-Audio":    "models.qwen2_audio.Qwen2AudioModel",
    "SALMONN-13B":    "models.salmonn_13b.SALMONNModel",
}


def list_models() -> list[str]:
    return list(_REGISTRY.keys())


def get_model(name: str, device: str = "cuda") -> AudioModel:
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise KeyError(f"Model '{name}' not found. Available: {available}")

    module_path, class_name = _REGISTRY[name].rsplit(".", 1)
    import importlib
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(device=device)


__all__ = ["AudioModel", "InferenceResult", "list_models", "get_model"]
