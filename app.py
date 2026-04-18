from __future__ import annotations

import io
import os
from pathlib import Path

import httpx
import matplotlib
matplotlib.use("Agg")  # must be called before pyplot import
import matplotlib.figure
import matplotlib.pyplot as plt
import soundfile as sf
import streamlit as st
import wandas

st.set_page_config(
    page_title="AudioLLMArena",
    page_icon="🎧",
    layout="wide",
)

_DEFAULT_BASE = os.environ.get("ARENA_API_BASE", "http://localhost")

MODEL_ENDPOINTS: dict[str, str] = {
    "Qwen2-Audio":    f"{_DEFAULT_BASE}:8600",
    "Audio Flamingo": f"{_DEFAULT_BASE}:8601",
    "Gemma-4-E4B":    f"{_DEFAULT_BASE}:8602",
    "MOSS-Audio-4B":  f"{_DEFAULT_BASE}:8603",
    "MOSS-Audio-8B":  f"{_DEFAULT_BASE}:8604",
    "SALMONN-13B":    f"{_DEFAULT_BASE}:8605",
}


def _check_health(base_url: str) -> bool:
    try:
        r = httpx.get(f"{base_url}/health", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def _run_inference(
    base_url: str,
    audio_bytes: bytes,
    filename: str,
    question: str,
    max_new_tokens: int,
) -> dict:
    with httpx.Client(timeout=300.0) as client:
        r = client.post(
            f"{base_url}/infer",
            files={"audio": (filename, audio_bytes, "audio/wav")},
            data={"question": question, "max_new_tokens": max_new_tokens},
        )
    r.raise_for_status()
    return r.json()


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
        options=list(MODEL_ENDPOINTS.keys()),
        default=list(MODEL_ENDPOINTS.keys())[:1],
    )
    max_new_tokens = st.slider("最大生成トークン数", min_value=64, max_value=1024, value=512, step=64)
    st.divider()
    st.subheader("可視化オプション")
    fmin = st.number_input("最低周波数 fmin (Hz)", min_value=0, max_value=20000, value=0, step=100)
    fmax_input = st.number_input("最高周波数 fmax (Hz, 0=自動)", min_value=0, max_value=20000, value=0, step=100)
    fmax: float | None = float(fmax_input) if fmax_input > 0 else None
    cmap = st.selectbox("カラーマップ", ["jet", "viridis", "magma", "inferno", "plasma"], index=0)
    apply_aw = st.checkbox("A特性補正 (Aw)", value=False)
    st.divider()
    st.subheader("コンテナ状態")
    for model_name, url in MODEL_ENDPOINTS.items():
        ok = _check_health(url)
        st.markdown(f"{'🟢' if ok else '🔴'} **{model_name}** `{url}`")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("🎧 AudioLLMArena")
st.caption("Native Audio-Language Models のオープンソース比較プレイグラウンド")

uploaded = st.file_uploader(
    "音声ファイルをアップロード（WAV / MP3 / FLAC / M4A / OGG）",
    type=["wav", "mp3", "flac", "m4a", "ogg"],
)

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

if uploaded is not None:
    audio_bytes = uploaded.read()
    uploaded.seek(0)

    if run and selected_models:
        cols = st.columns(len(selected_models))
        for col, model_name in zip(cols, selected_models):
            with col:
                st.markdown(f"### {model_name}")
                base_url = MODEL_ENDPOINTS[model_name]
                try:
                    with st.spinner(f"{model_name} 推論中…"):
                        result = _run_inference(
                            base_url=base_url,
                            audio_bytes=audio_bytes,
                            filename=uploaded.name,
                            question=question,
                            max_new_tokens=max_new_tokens,
                        )
                    st.success("完了")
                    st.write(result["answer"])
                    with st.expander("詳細"):
                        st.write(f"**モデル ID**: {result['model_id']}")
                        st.write(f"**推論時間**: {result['latency_ms']:.0f} ms")
                        st.write(f"**エンドポイント**: {base_url}")
                        st.write(f"**ファイル**: {uploaded.name}")
                        st.write(f"**質問**: {question}")
                except httpx.HTTPStatusError as exc:
                    st.error(f"API エラー {exc.response.status_code}: {exc.response.text}")
                except httpx.RequestError as exc:
                    st.error(f"接続エラー: {exc} — コンテナが起動しているか確認してください。")

    st.divider()
    st.subheader("プレビュー")
    st.audio(audio_bytes, format=f"audio/{Path(uploaded.name).suffix.lstrip('.')}")

    with st.spinner("波形・スペクトログラムを描画中…"):
        try:
            cf = _to_channel_frame(audio_bytes, uploaded.name)
            ylim: tuple | None = (
                (float(fmin), fmax) if (fmin > 0 or fmax is not None) else None
            )
            figs: list[matplotlib.figure.Figure] = cf.describe(  # type: ignore[assignment]
                is_close=False,
                fmin=float(fmin),
                fmax=fmax,
                ylim=ylim,
                cmap=cmap,
                Aw=apply_aw,
            )
            if figs:
                cols = st.columns(len(figs))
                for col, fig in zip(cols, figs):
                    with col:
                        st.pyplot(fig, use_container_width=False)
                        plt.close(fig)
        except Exception as exc:
            st.warning(f"可視化をスキップしました: {exc}")
