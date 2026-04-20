# AudioLLMArena
An Open-Source Benchmark & Playground for Native Audio-Language Models

AudioLLMArena は、次世代の「Native Audio-Language Models」をサイド・バイ・サイドで比較・評価するためのオープンソース・プレイグラウンドです。
ASR（文字起こし）を介さず、音声を直接「理解」するモデル（Native Audio LLMs）が、会話、環境音、感情、音楽をどのように推論するかを、開発者が直感的に検証・ベンチマークできる環境を提供します。

------------------------------

## 🚀 Key Features

* **Side-by-Side Comparison**: 同一の音声ファイルとプロンプトに対し、複数のAudio LLMが生成した回答を並べて比較。
* **Native Audio Understanding**: 音声信号を直接入力可能なモデル（Audio-Native）に特化。
* **Diverse Tasks**: 音声要約、感情分析、背景音の特定、音楽の解釈など、多角的なテストが可能。
* **Extensible Architecture**: 新しいオープンソースモデルや商用APIを簡単に追加できるプラグイン構造。

## 🎧 Supported Models

| モデル | パラメータ | 必要VRAM | HuggingFace |
|---|---|---|---|
| [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) | 7B | ~14 GB | Qwen/Qwen2-Audio-7B-Instruct |
| [Audio Flamingo Next](https://huggingface.co/nvidia/audio-flamingo-next-hf) | 7B | ~14 GB | nvidia/audio-flamingo-next-hf |
| [Audio Flamingo Next Captioner](https://huggingface.co/nvidia/audio-flamingo-next-captioner-hf) | 7B | ~14 GB | nvidia/audio-flamingo-next-captioner-hf |
| [Audio Flamingo Next Think](https://huggingface.co/nvidia/audio-flamingo-next-think-hf) | 7B | ~14 GB | nvidia/audio-flamingo-next-think-hf |
| [Gemma-4-E4B](https://huggingface.co/google/gemma-4-E4B-it) | 4B | ~8 GB | google/gemma-4-E4B-it |
| [MOSS-Audio-4B](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-4B-Instruct) | 4B | ~8 GB | OpenMOSS-Team/MOSS-Audio-4B-Instruct |
| [MOSS-Audio-8B](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-8B-Instruct) | 8B | ~16 GB | OpenMOSS-Team/MOSS-Audio-8B-Instruct |
| [MOSS-Audio-8B-Thinking](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-8B-Thinking) | 8B | ~16 GB | OpenMOSS-Team/MOSS-Audio-8B-Thinking |
| [SALMONN-13B](https://huggingface.co/tsinghua-ee/SALMONN) ⚠️ | 13B | ~26 GB | tsinghua-ee/SALMONN |
| [Step-Audio-R1.1](https://huggingface.co/stepfun-ai/Step-Audio-R1.1) | 33B | ~67 GB | stepfun-ai/Step-Audio-R1.1 |
| [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | 30B-A3B (MoE) | ~60 GB | Qwen/Qwen3-Omni-30B-A3B-Instruct |
| [Qwen3-Omni-Captioner](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner) | 30B-A3B (MoE) | ~60 GB | Qwen/Qwen3-Omni-30B-A3B-Captioner |
| [Qwen3-Omni-Thinking](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking) | 30B-A3B (MoE) | ~60 GB | Qwen/Qwen3-Omni-30B-A3B-Thinking |

> **Think / Thinking モデル**は推論過程（`<think>...</think>`）を UI 上の「思考過程」エキスパンダーで確認できます。

> ⚠️ **SALMONN-13B** は BEATs の事前学習済みチェックポイント（`BEATs_iter3_plus_AS2M.pt`）を手動でダウンロードして `SALMONN_BEATS_PATH` 環境変数に設定する必要があります。詳細は [起動方法](#3-streamlit-ui-の起動) を参照してください。

## 🛠 Installation

Python 3.11 と [uv](https://docs.astral.sh/uv/) が必要です。

> **HuggingFace キャッシュの場所**  
> デフォルトでは `/workspace/.cache/huggingface` にキャッシュが保存されます。  
> 大容量の外部ストレージを使いたい場合は `.devcontainer/devcontainer.json` のコメントアウトされたマウント行を参照してください。

```bash
# リポジトリのクローン
git clone https://github.com/kasahart/AudioLLMArena.git
cd AudioLLMArena

# 依存関係のインストール
uv sync --frozen
```

## 🏃‍♂️ Quick Start

AudioLLMArena は **各モデルを独立した Docker コンテナ**で動かし、Streamlit UI からコンテナの推論 API を呼び出す構成です。

### 1. 推論コンテナの起動

**GPU 環境（推奨）:**

```bash
# 全モデルを起動（大型モデルを除く）
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d \
  qwen2-audio audio-flamingo audio-flamingo-captioner audio-flamingo-think \
  gemma4-e4b moss-4b moss-8b moss-8b-thinking

# Qwen3-Omni（~60 GB VRAM 必要、3バリアント）
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d \
  qwen3-omni qwen3-omni-captioner qwen3-omni-thinking

# 特定のモデルだけ起動
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d qwen2-audio
```

**Step-Audio-R1.1（~67 GB VRAM 必要）:**

```bash
# vLLM サイドカー + FastAPI プロキシの2コンテナ構成
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d step-audio-vllm step-audio-r1
```

**SALMONN-13B（要 `SALMONN_BEATS_PATH` 環境変数）:**

```bash
SALMONN_BEATS_PATH=/path/to/BEATs_iter3_plus_AS2M.pt \
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile salmonn up -d salmonn-13b
```

### 2. サービス一覧

| サービス名 | モデル | ポート | 必要VRAM |
|---|---|---|---|
| `qwen2-audio` | Qwen2-Audio-7B | 8600 | ~14 GB |
| `audio-flamingo` | Audio Flamingo Next | 8601 | ~14 GB |
| `audio-flamingo-captioner` | Audio Flamingo Next Captioner | 8607 | ~14 GB |
| `audio-flamingo-think` | Audio Flamingo Next Think | 8608 | ~14 GB |
| `gemma4-e4b` | Gemma-4-E4B | 8602 | ~8 GB |
| `moss-4b` | MOSS-Audio-4B | 8603 | ~8 GB |
| `moss-8b` | MOSS-Audio-8B | 8604 | ~16 GB |
| `moss-8b-thinking` | MOSS-Audio-8B-Thinking | 8606 | ~16 GB |
| `salmonn-13b` | SALMONN-13B | 8605 | ~26 GB |
| `step-audio-vllm` + `step-audio-r1` | Step-Audio-R1.1 | 8609 | ~67 GB |
| `qwen3-omni` | Qwen3-Omni-30B-A3B (Instruct) | 8610 | ~60 GB |
| `qwen3-omni-captioner` | Qwen3-Omni-30B-A3B-Captioner | 8611 | ~60 GB |
| `qwen3-omni-thinking` | Qwen3-Omni-30B-A3B-Thinking | 8612 | ~60 GB |

コンテナの起動状態はヘルスチェックで確認できます。

```bash
docker compose ps
# または個別に確認
curl http://localhost:8600/health
```

### 3. Streamlit UI の起動

```bash
uv run streamlit run app.py
```

起動後、ブラウザで http://localhost:8501 にアクセスしてください。  
サイドバーに各コンテナの稼働状態（🟢/🔴）と必要 VRAM が表示されます。

> **コンテナのホストを変更する場合** は環境変数 `ARENA_API_BASE` で上書きできます。  
> 例: `ARENA_API_BASE=http://192.168.1.10 uv run streamlit run app.py`

## 📊 Evaluation Metrics

各モデルの推論結果とあわせて以下のメトリクスを UI 上で確認できます。

* **Inference Latency**: サーバー側での推論時間（ms）をレスポンスごとに表示

------------------------------

## 🤝 Contribution

新しいモデルの統合や、評価用データセットの提供を歓迎します。
特に、Audio Flamingo や Gemmaベースの最新モデルに最適化された推論アダプターのPRを歓迎します。

## 📜 License

MIT License
