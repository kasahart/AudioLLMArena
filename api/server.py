from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile

from models import get_model

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    name = os.environ.get("MODEL_NAME", "")
    if not name:
        raise RuntimeError("MODEL_NAME environment variable is required")
    device = os.environ.get("DEVICE", "cuda")
    model = get_model(name, device=device)
    model.load()
    app.state.model = model
    logger.info("Ready: %s (%s)", model.display_name, model.model_id)
    yield


app = FastAPI(title="AudioLLMArena Inference API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": app.state.model.display_name}


@app.get("/info")
def info():
    m = app.state.model
    return {"model_id": m.model_id, "display_name": m.display_name}


@app.post("/infer")
async def infer(
    audio: UploadFile,
    question: str = Form(...),
    max_new_tokens: int = Form(512),
):
    suffix = Path(audio.filename or "upload").suffix.lower() or ".wav"
    content = await audio.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: app.state.model.run_inference(tmp_path, question, max_new_tokens),
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)

    response: dict = {
        "answer": result.answer,
        "latency_ms": result.latency_ms,
        "model_id": result.model_id,
    }
    if result.thinking is not None:
        response["thinking"] = result.thinking
    return response
