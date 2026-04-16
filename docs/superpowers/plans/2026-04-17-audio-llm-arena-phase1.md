# AudioLLMArena Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audio Flamingo モデルを Streamlit UI で動かせる状態にし、将来のモデル追加が `models/` に1ファイル追加するだけで完結するプラグイン構造を実装する。

**Architecture:** `models/base.py` に ABC (`AudioModel`) と結果型 (`InferenceResult`) を定義し、`models/audio_flamingo.py` がそれを実装するアダプターパターン。`models/__init__.py` のレジストリ辞書が UI とモデル層を疎結合にする。`app.py` はレジストリ経由でモデルを取得し `st.cache_resource` でセッション中に常駐させる。

**Tech Stack:** Python 3.11, Streamlit, wandas, soundfile, PyTorch (CUDA 12.8), transformers (custom build: nvidia/audio-flamingo-next-hf), pytest

---

## ファイルマップ

| 操作 | パス | 責務 |
|---|---|---|
| 作成 | `models/__init__.py` | レジストリ・`list_models()`・`get_model()` |
| 作成 | `models/base.py` | `InferenceResult` dataclass・`AudioModel` ABC |
| 作成 | `models/audio_flamingo.py` | Audio Flamingo アダプター実装 |
| 作成 | `app.py` | Streamlit UI |
| 作成 | `tests/__init__.py` | テストパッケージ |
| 作成 | `tests/test_base.py` | base.py のテスト |
| 作成 | `tests/test_audio_flamingo.py` | audio_flamingo.py のテスト |
| 作成 | `tests/test_registry.py` | registry のテスト |
| 修正 | `pyproject.toml` | pytest を dev 依存に追加 |

---

## Task 1: pytest を dev 依存に追加

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: pyproject.toml に dev 依存セクションを追加**

`pyproject.toml` の `[tool.uv]` セクションの前に以下を追加する:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
]
```

- [ ] **Step 2: dev 依存をインストール**

```bash
uv sync --extra dev
```

期待される出力（抜粋）:
```
+ pytest==8.x.x
```

- [ ] **Step 3: pytest が動くことを確認**

```bash
uv run pytest --version
```

期待される出力例: `pytest 8.x.x`

- [ ] **Step 4: コミット**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add pytest as dev dependency"
```

---

## Task 2: models/base.py — InferenceResult と AudioModel ABC

**Files:**
- Create: `models/__init__.py` (空ファイル、パッケージ化のため)
- Create: `models/base.py`
- Create: `tests/__init__.py`
- Create: `tests/test_base.py`

- [ ] **Step 1: models/ パッケージを空の `__init__.py` で初期化**

`models/__init__.py` は Task 3 で上書きするため、今は空ファイルを作成する:

```bash
mkdir -p models tests
touch models/__init__.py tests/__init__.py
```

- [ ] **Step 2: `models/base.py` を作成**

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class InferenceResult:
    answer: str
    latency_ms: float
    model_id: str


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
```

- [ ] **Step 3: テストを書く**

`tests/test_base.py`:

```python
from __future__ import annotations

import pytest
from pathlib import Path

from models.base import AudioModel, InferenceResult


def test_inference_result_stores_fields():
    result = InferenceResult(answer="hello", latency_ms=123.4, model_id="test/model")
    assert result.answer == "hello"
    assert result.latency_ms == 123.4
    assert result.model_id == "test/model"


def test_audio_model_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        AudioModel()  # type: ignore[abstract]


class _ConcreteModel(AudioModel):
    def load(self) -> None:
        pass

    def run_inference(self, audio_path: Path, question: str, max_new_tokens: int = 512) -> InferenceResult:
        return InferenceResult(answer="ok", latency_ms=1.0, model_id="test/model")

    @property
    def display_name(self) -> str:
        return "Test Model"

    @property
    def model_id(self) -> str:
        return "test/model"


def test_concrete_subclass_satisfies_interface():
    m = _ConcreteModel()
    assert m.display_name == "Test Model"
    assert m.model_id == "test/model"


def test_concrete_subclass_run_inference_returns_result(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fake")
    m = _ConcreteModel()
    result = m.run_inference(audio, "What is this?")
    assert isinstance(result, InferenceResult)
    assert result.answer == "ok"
```

- [ ] **Step 4: テストを実行して PASS を確認**

```bash
uv run pytest tests/test_base.py -v
```

期待される出力（抜粋）:
```
tests/test_base.py::test_inference_result_stores_fields PASSED
tests/test_base.py::test_audio_model_cannot_be_instantiated_directly PASSED
tests/test_base.py::test_concrete_subclass_satisfies_interface PASSED
tests/test_base.py::test_concrete_subclass_run_inference_returns_result PASSED
4 passed
```

- [ ] **Step 5: コミット**

```bash
git add models/__init__.py models/base.py tests/__init__.py tests/test_base.py
git commit -m "feat: add AudioModel ABC and InferenceResult dataclass"
```

---

## Task 3: models/audio_flamingo.py — Audio Flamingo アダプター

**Files:**
- Create: `models/audio_flamingo.py`
- Create: `tests/test_audio_flamingo.py`

- [ ] **Step 1: テストを書く（先に失敗を確認する）**

`tests/test_audio_flamingo.py`:

```python
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from models.audio_flamingo import AudioFlamingoModel
from models.base import InferenceResult


def test_display_name():
    m = AudioFlamingoModel()
    assert m.display_name == "Audio Flamingo"


def test_model_id():
    m = AudioFlamingoModel()
    assert m.model_id == "nvidia/audio-flamingo-next-hf"


def test_load_is_idempotent():
    m = AudioFlamingoModel(device="cpu")

    with patch("models.audio_flamingo.AutoModel") as MockModel, \
         patch("models.audio_flamingo.AutoProcessor") as MockProcessor, \
         patch("models.audio_flamingo.torch") as mock_torch:

        mock_torch.cuda.is_available.return_value = False
        mock_torch.float32 = "float32"
        MockModel.from_pretrained.return_value = MagicMock()
        MockProcessor.from_pretrained.return_value = MagicMock()

        m.load()
        m.load()  # 2回目は早期リターンするため from_pretrained は1回だけ

        assert MockModel.from_pretrained.call_count == 1


def test_load_falls_back_to_cpu_when_cuda_unavailable():
    m = AudioFlamingoModel(device="cuda")

    with patch("models.audio_flamingo.AutoModel") as MockModel, \
         patch("models.audio_flamingo.AutoProcessor") as MockProcessor, \
         patch("models.audio_flamingo.torch") as mock_torch:

        mock_torch.cuda.is_available.return_value = False
        mock_torch.float32 = "float32"
        MockModel.from_pretrained.return_value = MagicMock()
        MockProcessor.from_pretrained.return_value = MagicMock()

        m.load()

        assert m._device == "cpu"


def test_load_raises_runtime_error_on_unsupported_transformers():
    m = AudioFlamingoModel(device="cpu")

    with patch("models.audio_flamingo.AutoModel") as MockModel, \
         patch("models.audio_flamingo.AutoProcessor") as MockProcessor, \
         patch("models.audio_flamingo.torch") as mock_torch:

        mock_torch.cuda.is_available.return_value = False
        mock_torch.float32 = "float32"
        MockProcessor.from_pretrained.return_value = MagicMock()
        MockModel.from_pretrained.side_effect = ValueError("audioflamingonext not supported")

        with pytest.raises(RuntimeError, match="transformers build"):
            m.load()


def test_run_inference_raises_on_unsupported_extension(tmp_path):
    audio_file = tmp_path / "audio.xyz"
    audio_file.write_bytes(b"fake")

    m = AudioFlamingoModel()
    m._model = MagicMock()
    m._processor = MagicMock()
    m._device = "cpu"

    with pytest.raises(ValueError, match="Unsupported audio format"):
        m.run_inference(audio_file, "What do you hear?")


def test_run_inference_returns_inference_result(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"fake wav")

    mock_input_ids = MagicMock()
    mock_input_ids.shape = (1, 10)
    mock_batch = {"input_ids": mock_input_ids}

    mock_apply_result = MagicMock()
    mock_apply_result.to.return_value = mock_batch

    mock_generated = MagicMock()

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.dtype = MagicMock()
    mock_model.generate.return_value = mock_generated

    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = mock_apply_result
    mock_processor.batch_decode.return_value = ["This is the answer"]

    m = AudioFlamingoModel()
    m._model = mock_model
    m._processor = mock_processor
    m._device = "cpu"

    with patch("models.audio_flamingo.torch") as mock_torch:
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=None)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_torch.inference_mode.return_value = mock_ctx

        result = m.run_inference(audio_file, "What do you hear?")

    assert isinstance(result, InferenceResult)
    assert result.answer == "This is the answer"
    assert result.model_id == "nvidia/audio-flamingo-next-hf"
    assert result.latency_ms >= 0
```

- [ ] **Step 2: テストが FAIL することを確認（実装前）**

```bash
uv run pytest tests/test_audio_flamingo.py -v 2>&1 | head -20
```

期待される出力: `ModuleNotFoundError: No module named 'models.audio_flamingo'`

- [ ] **Step 3: `models/audio_flamingo.py` を実装**

```python
from __future__ import annotations

import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoProcessor

from models.base import AudioModel, InferenceResult

_MODEL_ID = "nvidia/audio-flamingo-next-hf"
_SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


class AudioFlamingoModel(AudioModel):
    def __init__(self, device: str = "cuda") -> None:
        self._requested_device = device
        self._device: str | None = None
        self._model = None
        self._processor = None

    @property
    def display_name(self) -> str:
        return "Audio Flamingo"

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
        self._processor = AutoProcessor.from_pretrained(_MODEL_ID, trust_remote_code=True)

        try:
            self._model = AutoModel.from_pretrained(
                _MODEL_ID,
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
        ).to(self._model.device)

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
        completion_ids = generated_ids[:, prompt_length:]
        answer = self._processor.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return InferenceResult(
            answer=answer,
            latency_ms=latency_ms,
            model_id=_MODEL_ID,
        )
```

- [ ] **Step 4: テストを実行して PASS を確認**

```bash
uv run pytest tests/test_audio_flamingo.py -v
```

期待される出力（抜粋）:
```
tests/test_audio_flamingo.py::test_display_name PASSED
tests/test_audio_flamingo.py::test_model_id PASSED
tests/test_audio_flamingo.py::test_load_is_idempotent PASSED
tests/test_audio_flamingo.py::test_load_falls_back_to_cpu_when_cuda_unavailable PASSED
tests/test_audio_flamingo.py::test_load_raises_runtime_error_on_unsupported_transformers PASSED
tests/test_audio_flamingo.py::test_run_inference_raises_on_unsupported_extension PASSED
tests/test_audio_flamingo.py::test_run_inference_returns_inference_result PASSED
7 passed
```

- [ ] **Step 5: コミット**

```bash
git add models/audio_flamingo.py tests/test_audio_flamingo.py
git commit -m "feat: implement AudioFlamingoModel adapter"
```

---

## Task 4: models/__init__.py — レジストリとヘルパー

**Files:**
- Modify: `models/__init__.py` (Task 2 で作成した空ファイルを上書き)
- Create: `tests/test_registry.py`

- [ ] **Step 1: テストを書く**

`tests/test_registry.py`:

```python
from __future__ import annotations

import pytest

from models import get_model, list_models
from models.audio_flamingo import AudioFlamingoModel


def test_list_models_contains_audio_flamingo():
    assert "Audio Flamingo" in list_models()


def test_get_model_returns_audio_flamingo_instance():
    m = get_model("Audio Flamingo", device="cpu")
    assert isinstance(m, AudioFlamingoModel)


def test_get_model_passes_device_to_instance():
    m = get_model("Audio Flamingo", device="cpu")
    assert m._requested_device == "cpu"


def test_get_model_raises_on_unknown_model():
    with pytest.raises(KeyError, match="Unknown Model"):
        get_model("Unknown Model")
```

- [ ] **Step 2: テストが FAIL することを確認**

```bash
uv run pytest tests/test_registry.py -v 2>&1 | head -20
```

期待される出力: エラー（`models/__init__.py` が空のため `get_model` が存在しない）

- [ ] **Step 3: `models/__init__.py` を実装**

```python
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
```

- [ ] **Step 4: テストを実行して PASS を確認**

```bash
uv run pytest tests/test_registry.py -v
```

期待される出力:
```
tests/test_registry.py::test_list_models_contains_audio_flamingo PASSED
tests/test_registry.py::test_get_model_returns_audio_flamingo_instance PASSED
tests/test_registry.py::test_get_model_passes_device_to_instance PASSED
tests/test_registry.py::test_get_model_raises_on_unknown_model PASSED
4 passed
```

- [ ] **Step 5: 全テストスイートを実行**

```bash
uv run pytest tests/ -v
```

期待される出力: 全テスト PASS（`test_base`, `test_audio_flamingo`, `test_registry` を含む）

- [ ] **Step 6: コミット**

```bash
git add models/__init__.py tests/test_registry.py
git commit -m "feat: add model registry with list_models and get_model helpers"
```

---

## Task 5: app.py — Streamlit UI

**Files:**
- Create: `app.py`

Streamlit アプリは単体テスト不可のため、実装後に手動で起動して動作を確認する。

- [ ] **Step 1: `app.py` を作成**

```python
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import soundfile as sf
import streamlit as st
import wandas

from models import get_model, list_models

matplotlib.use("Agg")

st.set_page_config(
    page_title="AudioLLMArena",
    page_icon="🎧",
    layout="wide",
)


@st.cache_resource(show_spinner="モデルをロード中…（初回のみ）")
def _cached_model(name: str, device: str):
    m = get_model(name, device=device)
    m.load()
    return m


def _to_channel_frame(data: bytes, filename: str) -> wandas.ChannelFrame:
    suffix = Path(filename).suffix.lower()
    stem = Path(filename).stem

    if suffix == ".wav":
        buf = io.BytesIO(data)
        buf.name = filename
        return wandas.read_wav(buf)

    audio_np, sr = sf.read(io.BytesIO(data), always_2d=True, dtype="float32")
    return wandas.from_ndarray(audio_np.T, sampling_rate=sr, frame_label=stem)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("設定")
    selected_models = st.multiselect(
        "モデル",
        options=list_models(),
        default=list_models()[:1],
    )
    device_choice = st.selectbox("デバイス", ["cuda", "cpu"], index=0)
    max_new_tokens = st.slider("最大生成トークン数", min_value=64, max_value=1024, value=512, step=64)
    st.divider()
    st.subheader("可視化オプション")
    fmin = st.number_input("最低周波数 fmin (Hz)", min_value=0, max_value=20000, value=0, step=100)
    fmax_input = st.number_input("最高周波数 fmax (Hz, 0=自動)", min_value=0, max_value=20000, value=0, step=100)
    fmax: float | None = float(fmax_input) if fmax_input > 0 else None
    cmap = st.selectbox("カラーマップ", ["jet", "viridis", "magma", "inferno", "plasma"], index=0)
    apply_aw = st.checkbox("A特性補正 (Aw)", value=False)
    st.divider()
    st.caption("モデルは初回リクエスト時に一度だけロードされ、以降はメモリに常駐します。")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("🎧 AudioLLMArena")
st.caption("Native Audio-Language Models のオープンソース比較プレイグラウンド")

uploaded = st.file_uploader(
    "音声ファイルをアップロード（WAV / MP3 / FLAC / M4A / OGG）",
    type=["wav", "mp3", "flac", "m4a", "ogg"],
)

if uploaded is not None:
    audio_bytes = uploaded.read()
    uploaded.seek(0)

    st.subheader("プレビュー")
    st.audio(audio_bytes, format=f"audio/{Path(uploaded.name).suffix.lstrip('.')}")

    with st.spinner("波形・スペクトログラムを描画中…"):
        try:
            cf = _to_channel_frame(audio_bytes, uploaded.name)
            ylim: tuple | None = (
                (float(fmin), fmax) if (fmin > 0 or fmax is not None) else None
            )
            figs: list = cf.describe(
                is_close=False,
                fmin=float(fmin),
                fmax=fmax,
                ylim=ylim,
                cmap=cmap,
                Aw=apply_aw,
            )
            for i, fig in enumerate(figs):
                if len(figs) > 1:
                    st.caption(f"チャンネル {i + 1}")
                st.pyplot(fig)
                plt.close(fig)
        except Exception as exc:
            st.warning(f"可視化をスキップしました: {exc}")

    st.divider()

st.subheader("推論")
question = st.text_input(
    "質問",
    value="What do you hear in this audio?",
    placeholder="例: この音声は何ですか？",
)

if not selected_models:
    st.info("サイドバーでモデルを1つ以上選択してください。")

run = st.button(
    "推論を実行",
    type="primary",
    disabled=(uploaded is None or not selected_models),
)

if run and uploaded is not None and selected_models:
    suffix = Path(uploaded.name).suffix.lower()
    uploaded.seek(0)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = Path(tmp.name)

    try:
        cols = st.columns(len(selected_models))
        for col, model_name in zip(cols, selected_models):
            with col:
                st.markdown(f"### {model_name}")
                try:
                    model = _cached_model(model_name, device_choice)
                    with st.spinner(f"{model_name} 推論中…"):
                        result = model.run_inference(
                            audio_path=tmp_path,
                            question=question,
                            max_new_tokens=max_new_tokens,
                        )
                    st.success("完了")
                    st.write(result.answer)
                    with st.expander("詳細"):
                        st.write(f"**モデル ID**: {result.model_id}")
                        st.write(f"**推論時間**: {result.latency_ms:.0f} ms")
                        st.write(f"**デバイス**: {device_choice}")
                        st.write(f"**ファイル**: {uploaded.name}")
                        st.write(f"**質問**: {question}")
                except (ValueError, RuntimeError) as exc:
                    st.error(f"エラー: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)
```

- [ ] **Step 2: Python の構文チェック**

```bash
uv run python -c "import ast; ast.parse(open('app.py').read()); print('syntax OK')"
```

期待される出力: `syntax OK`

- [ ] **Step 3: import が通ることを確認（Streamlit なしで）**

```bash
uv run python -c "
import sys
sys.modules['streamlit'] = __import__('types').ModuleType('streamlit')
# app.py の非 Streamlit import だけ確認
from models import get_model, list_models
from models.audio_flamingo import AudioFlamingoModel
print('imports OK')
"
```

期待される出力: `imports OK`

- [ ] **Step 4: コミット**

```bash
git add app.py
git commit -m "feat: add Streamlit UI for AudioLLMArena"
```

---

## Task 6: 全テスト実行と最終コミット

**Files:** なし（確認のみ）

- [ ] **Step 1: 全テストスイートを実行**

```bash
uv run pytest tests/ -v
```

期待される出力:
```
tests/test_base.py::test_inference_result_stores_fields PASSED
tests/test_base.py::test_audio_model_cannot_be_instantiated_directly PASSED
tests/test_base.py::test_concrete_subclass_satisfies_interface PASSED
tests/test_base.py::test_concrete_subclass_run_inference_returns_result PASSED
tests/test_audio_flamingo.py::test_display_name PASSED
tests/test_audio_flamingo.py::test_model_id PASSED
tests/test_audio_flamingo.py::test_load_is_idempotent PASSED
tests/test_audio_flamingo.py::test_load_falls_back_to_cpu_when_cuda_unavailable PASSED
tests/test_audio_flamingo.py::test_load_raises_runtime_error_on_unsupported_transformers PASSED
tests/test_audio_flamingo.py::test_run_inference_raises_on_unsupported_extension PASSED
tests/test_audio_flamingo.py::test_run_inference_returns_inference_result PASSED
tests/test_registry.py::test_list_models_contains_audio_flamingo PASSED
tests/test_registry.py::test_get_model_returns_audio_flamingo_instance PASSED
tests/test_registry.py::test_get_model_passes_device_to_instance PASSED
tests/test_registry.py::test_get_model_raises_on_unknown_model PASSED
15 passed
```

- [ ] **Step 2: ファイル構成を確認**

```bash
find . -name "*.py" -not -path "./venv/*" -not -path "./.git/*" | sort
```

期待される出力（抜粋）:
```
./app.py
./models/__init__.py
./models/audio_flamingo.py
./models/base.py
./tests/__init__.py
./tests/test_audio_flamingo.py
./tests/test_base.py
./tests/test_registry.py
```

- [ ] **Step 3: 最終コミット**

```bash
git add -A
git status  # 未追跡ファイルがないことを確認
git commit -m "feat: AudioLLMArena Phase 1 complete — Audio Flamingo with plugin architecture"
```
