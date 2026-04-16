# AudioLLMArena — 実装設計仕様書

**日付**: 2026-04-17  
**フェーズ**: Phase 1 — Audio Flamingo のみ対応  
**目標**: Native Audio-Language Model をブラウザで試せる Streamlit アプリを、将来のマルチモデル比較に向けたプラグイン構造で実装する

---

## 1. ファイル構成

```
AudioLLMArena/
├── app.py                        # Streamlit UI（メインエントリ）
├── models/
│   ├── __init__.py               # レジストリ定義・get_model() ヘルパー
│   ├── base.py                   # AudioModel ABC + InferenceResult dataclass
│   └── audio_flamingo.py         # Audio Flamingo アダプター
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-04-17-audio-llm-arena-design.md
```

---

## 2. データ型・インターフェース

### `models/base.py`

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

@dataclass
class InferenceResult:
    answer: str          # モデルの出力テキスト
    latency_ms: float    # 推論時間（ms）— TTFT 相当
    model_id: str        # Hugging Face モデル ID

class AudioModel(ABC):
    @abstractmethod
    def load(self) -> None:
        """モデルとプロセッサをロードする。冪等であること。"""

    @abstractmethod
    def run_inference(
        self,
        audio_path: Path,
        question: str,
        max_new_tokens: int = 512,
    ) -> InferenceResult:
        """音声ファイルと質問から推論結果を返す。"""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """UI 表示用のモデル名。"""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Hugging Face モデル ID。"""
```

### `models/__init__.py`

```python
from models.base import AudioModel, InferenceResult
from models.audio_flamingo import AudioFlamingoModel

REGISTRY: dict[str, type[AudioModel]] = {
    "Audio Flamingo": AudioFlamingoModel,
}

def list_models() -> list[str]:
    return list(REGISTRY.keys())

def get_model(name: str, device: str = "cuda") -> AudioModel:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model: {name}")
    return REGISTRY[name](device=device)
```

新モデルは `REGISTRY` に1行追加するだけで UI に反映される。

---

## 3. Audio Flamingo アダプター

### `models/audio_flamingo.py`

参考実装（`kasahart/audio-flamingo-next-env`）のロジックを `AudioModel` インターフェースに移植する。

- **モデル ID**: `nvidia/audio-flamingo-next-hf`
- **`load()`**: `AutoProcessor` + `AutoModel` をロード。`bfloat16`（CUDA）または `float32`（CPU）。`device_map="auto"`（CUDA のみ）。すでにロード済みの場合は早期リターン（冪等）。
- **`run_inference()`**: chat template ベースで `processor.apply_chat_template()` → `model.generate()` → デコード。経過時間を `time.perf_counter()` で計測して `latency_ms` に格納。
- **`display_name`**: `"Audio Flamingo"`
- **`model_id`**: `"nvidia/audio-flamingo-next-hf"`

---

## 4. Streamlit UI（`app.py`）

### サイドバー

| 設定項目 | ウィジェット | デフォルト |
|---|---|---|
| モデル選択 | `st.multiselect`（`list_models()` から選択肢生成） | `["Audio Flamingo"]` |
| デバイス | `st.selectbox(["cuda", "cpu"])` | `"cuda"` |
| 最大生成トークン | `st.slider(64–1024, step=64)` | `512` |
| fmin / fmax | `st.number_input` | `0` / 自動 |
| カラーマップ | `st.selectbox` | `"jet"` |
| A特性補正 | `st.checkbox` | `False` |

### メインエリアのフロー

```
1. 音声アップロード (wav/mp3/flac/m4a/ogg)
   └─ st.audio でプレビュー
   └─ wandas.ChannelFrame → cf.describe() で波形・スペクトログラム表示

2. 質問入力 (st.text_input)

3. 「推論を実行」ボタン
   └─ 選択されたモデルごとに load() + run_inference() を順次実行
   └─ st.columns(n_models) で並列パネル表示
      各パネル: 回答テキスト / 推論時間 / モデル ID / 質問

4. エラーは st.error で表示、一時ファイルは finally で削除
```

### モデルキャッシュ戦略

`st.cache_resource` でモデルインスタンスをキャッシュ。キャッシュキーは `(model_name, device)`。

```python
@st.cache_resource(show_spinner="モデルをロード中…（初回のみ）")
def _cached_model(name: str, device: str) -> AudioModel:
    m = get_model(name, device=device)
    m.load()
    return m
```

---

## 5. エラーハンドリング方針

| 状況 | 対応 |
|---|---|
| 非対応の音声形式 | `ValueError` を raise → `st.error` で表示 |
| CUDA 不在 | `AudioFlamingoModel.load()` 内で CUDA 可用性を確認し、自動 CPU フォールバック + `st.warning` 表示 |
| モデルロード失敗（transformers 非対応ビルド等） | `RuntimeError` を raise → `st.error` で表示 |
| 可視化失敗 | `st.warning` でスキップ（推論は継続） |

---

## 6. 対象外（Phase 1 スコープ外）

- 自動評価メトリクス（Contextual Accuracy, Emotion Recognition）の算出
- Audio Flamingo 以外のモデル実装
- バッチ評価 / データセット読み込み
- 結果の保存・エクスポート

---

## 7. 依存関係

既存の `pyproject.toml` の依存で全て賄える。追加パッケージなし。

- `streamlit` — UI
- `wandas` — 音声可視化
- `soundfile` — 非 WAV フォーマットデコード
- `transformers`（カスタムビルド）— Audio Flamingo モデルロード
- `torch` / `torchaudio` — 推論バックエンド
- `matplotlib` — スペクトログラム描画

---

## 8. 将来のモデル追加手順（設計意図の記録）

1. `models/<new_model>.py` を作成し `AudioModel` を継承
2. `models/__init__.py` の `REGISTRY` に1行追加
3. UI・`base.py` の変更は不要
