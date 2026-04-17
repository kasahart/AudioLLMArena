from __future__ import annotations

from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "gpu: requires a CUDA GPU — skipped automatically when unavailable",
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        gpu_available = False
    if gpu_available:
        return
    skip = pytest.mark.skip(reason="GPU not available (torch.cuda.is_available() == False)")
    for item in items:
        if item.get_closest_marker("gpu"):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def sample_wav() -> Path:
    """Path to a committed 1-second 16 kHz sine-wave WAV file."""
    path = Path(__file__).parent / "fixtures" / "sample.wav"
    assert path.exists(), f"Test fixture not found: {path}"
    return path
