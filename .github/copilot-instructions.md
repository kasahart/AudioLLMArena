# LALMsArena — Copilot Instructions

> モデル一覧・VRAM・起動コマンドは `README.md` を参照。ここでは開発・実装上のノウハウを記載する。

---

## アーキテクチャ

### API サーバー（`api/server.py`）
- FastAPI + uvicorn で動作
- 起動時に `MODEL_NAME` 環境変数を読み取り、`models/get_model()` でモデルをロードして `app.state.model` に保持
- エンドポイント：`GET /health`、`GET /info`、`POST /infer`
- `POST /infer`：`multipart/form-data` で `audio`（ファイル）と `question`（テキスト）を受け取る

### モデルクラス（`models/<model>.py`）
- 全モデルが `models/base.AudioModel` を継承
- 必須実装：`load()`、`run_inference(audio_path, question, max_new_tokens) -> InferenceResult`、`display_name`、`model_id`
- `InferenceResult`：`answer`、`latency_ms`、`model_id`、`thinking`（Optional）

### コンテナ構成（`containers/<model>/`）
- `Dockerfile`：`nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` ベース
- ビルド時に `/opt/<model>/.venv` を uv で作成（ホスト側 bind mount に依存しない）
- `VIRTUAL_ENV=/opt/<model>/.venv` と `PATH` を設定して venv を有効化
- `CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]`

---

## Docker Compose の注意点

### 重要：GPU コンテナは必ず両方の compose ファイルを指定する

```bash
# 正しい（GPU 割り当てあり）
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d <service>

# 誤り（GPU なしで起動→大規模モデルが OOM kill される）
docker compose up -d <service>
```

`docker-compose.gpu.yml` は `runtime: nvidia` と `DEVICE: cuda` を付与する。忘れると NVIDIA Driver not detected 警告が出て、30B 超モデルは CPU メモリ枯渇で exit 137 (SIGKILL) になる。

### サイドカーが必要なサービス
`step-audio-r1` は `step-audio-vllm` サイドカーが必要：
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d step-audio-vllm step-audio-r1
```
`nemotron-omni-reasoning` も同様に `nemotron-vllm` が必要。

---

## 新規コンテナを追加するときの手順

1. `containers/<model>/pyproject.toml` と `containers/<model>/uv.lock` を作成
   - `uv.lock` は必ず `uv lock --project containers/<model>` で生成する
   - pyproject.toml の dependencies を変更した後も再度 `uv lock` が必要
2. `containers/<model>/Dockerfile` を作成（下記テンプレート参照）
3. `docker-compose.yml` にサービスを追加（ポートとして 8600 番台を使用）
4. `docker-compose.gpu.yml` に `<<: *gpu` エントリを追加
5. `models/<model>.py` に `AudioModel` サブクラスを実装
6. `models/__init__.py` の `get_model()` に登録
7. `tests/test_<model>_api.py` に API 統合テストを作成

### Dockerfile テンプレート
```dockerfile
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git git-lfs curl ffmpeg libsndfile1 libsox-fmt-all sox build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python3 -m pip install --upgrade pip uv

COPY containers/<model>/pyproject.toml /opt/<model>/pyproject.toml
COPY containers/<model>/uv.lock        /opt/<model>/uv.lock

ENV VIRTUAL_ENV=/opt/<model>/.venv \
    PATH="/opt/<model>/.venv/bin:$PATH"

RUN cd /opt/<model> && uv sync --frozen --no-dev

WORKDIR /workspace

ENV PYTHONPATH=/workspace \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_CACHE=/root/.cache/huggingface/hub

EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## テスト

### API 統合テスト（`tests/test_<model>_api.py`）
コンテナが起動していない場合は自動スキップするパターン：

```python
def _is_up(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=3.0).status_code == 200
    except Exception:
        return False

@pytest.fixture(scope="module")
def base_url():
    if not _is_up(_BASE_URL):
        pytest.skip("container not running")
    return _BASE_URL
```

4 つの標準テスト：`test_health`、`test_info`、`test_infer_returns_answer`、`test_infer_rejects_unsupported_format`

複数バリアントがある場合は `@pytest.fixture(params=["instruct","captioner","thinking"])` でパラメータ化する。

### テスト実行：1 コンテナずつ逐次（VRAM 節約）
```bash
# 起動 → health 待機 → テスト → 停止
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d <service>
until curl -fsS http://localhost:<port>/health; do sleep 20; done
uv run --project /workspace pytest tests/test_<model>_api.py -v
docker compose -f docker-compose.yml -f docker-compose.gpu.yml stop <service>
```

### ユニットテスト
```bash
uv run pytest tests/ -v
```

---

## 依存関係の管理（uv）

```bash
# ホスト側（UI・テスト用）
uv sync

# コンテナ用 venv をホスト上でセットアップ（必要な場合）
UV_PROJECT_ENVIRONMENT=/workspace/containers/<model>/.venv \
  uv sync --frozen --project containers/<model> --python python3.11

# pyproject.toml 変更後は必ず再ロック
uv lock --project containers/<model>
```

`uv.lock` を更新した後はコンテナイメージを再ビルドする：
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build <service>
```

---

## GPU 環境

- **GPU 0**：NVIDIA RTX PRO 6000 Blackwell Max-Q（約 98 GB VRAM）
- **GPU 1**：NVIDIA T400 4GB
- 全コンテナはデフォルトで `CUDA_VISIBLE_DEVICES=0`（大型 GPU）を使用
- 複数コンテナを同時起動すると VRAM が不足するため、テスト時は 1 コンテナずつ起動・停止する

---

## よくある問題

| 症状 | 原因 | 対処 |
|---|---|---|
| コンテナが exit 137 で終了 | GPU なしで大規模モデルをロード→ OOM | `docker-compose.gpu.yml` を `-f` で指定して再起動 |
| `uvicorn: not found` (exit 127) | コンテナイメージが古い | `docker compose build --no-cache <service>` で再ビルド |
| `uv sync` が依存関係エラー | `uv.lock` に不足パッケージがある | `uv lock --project containers/<model>` を実行してから再ビルド |
| health エンドポイントが長時間 starting | モデルのダウンロード/ロード中 | 30B 超モデルは初回数分〜十数分かかる。ログで進捗を確認 |
| `SALMONN_BEATS_PATH` 警告 | BEATs モデルのパスが未設定 | `salmonn-13b` 起動時のみ必須。他サービスは無視して OK |
