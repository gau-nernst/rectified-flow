from __future__ import annotations

import io
import json
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from modelling import Flux2Pipeline, load_flux2

APP_ROOT = Path(__file__).resolve().parent
STATIC_DIR = APP_ROOT / "static"
TEMP_DIR = APP_ROOT / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
TEMP_INDEX_PATH = TEMP_DIR / "index.json"

MODELS = ["klein-4B", "klein-9B", "klein-base-4B", "klein-base-9B"]
MODEL_DEFAULTS = {
    "klein-4B": dict(guidance=None, cfg_scale=1.0, num_steps=4),
    "klein-9B": dict(guidance=None, cfg_scale=1.0, num_steps=4),
    "klein-base-4B": dict(guidance=None, cfg_scale=4.0, num_steps=50),
    "klein-base-9B": dict(guidance=None, cfg_scale=4.0, num_steps=50),
}

_PIPELINE: Flux2Pipeline | None = None
_ACTIVE_MODEL: str | None = None
_TEMP_INDEX: dict[str, dict[str, Any]] = {}


def _load_temp_index() -> None:
    global _TEMP_INDEX
    if TEMP_INDEX_PATH.exists():
        try:
            _TEMP_INDEX = json.loads(TEMP_INDEX_PATH.read_text())
        except json.JSONDecodeError:
            _TEMP_INDEX = {}


def _save_temp_index() -> None:
    TEMP_INDEX_PATH.write_text(json.dumps(_TEMP_INDEX, indent=2))


def _save_temp_image(img: Image.Image, label: str | None = None) -> str:
    temp_id = uuid.uuid4().hex
    filename = f"{temp_id}.png"
    path = TEMP_DIR / filename
    img.save(path, format="PNG")
    _TEMP_INDEX[temp_id] = {
        "id": temp_id,
        "filename": filename,
        "label": label or "",
        "created_at": time.time(),
        "width": img.width,
        "height": img.height,
    }
    _save_temp_index()
    return temp_id


def _get_temp_image(temp_id: str) -> Image.Image:
    info = _TEMP_INDEX.get(temp_id)
    if not info:
        raise HTTPException(status_code=404, detail="Unknown temp image")
    path = TEMP_DIR / info["filename"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Temp image missing on disk")
    return Image.open(path)


def _validate_img_size(width: int, height: int) -> tuple[int, int]:
    # Flux.2 AE path expects sizes that are multiples of 16
    width = max(16, (width // 16) * 16)
    height = max(16, (height // 16) * 16)
    return width, height


def _load_model(model_name: str) -> Flux2Pipeline:
    if model_name not in MODELS:
        raise HTTPException(status_code=400, detail="Unsupported model")
    if not torch.cuda.is_available():
        raise HTTPException(status_code=500, detail="CUDA is required for Flux.2 UI")

    global _PIPELINE, _ACTIVE_MODEL
    if _PIPELINE is not None:
        _PIPELINE.cpu()
        del _PIPELINE
        torch.cuda.empty_cache()

    flux = load_flux2(model_name)
    pipeline = Flux2Pipeline(flux=flux).cuda()
    _PIPELINE = pipeline
    _ACTIVE_MODEL = model_name
    return pipeline


def _get_pipeline() -> Flux2Pipeline:
    if _PIPELINE is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    return _PIPELINE


def _tensor_to_pil(img: torch.Tensor) -> Image.Image:
    if img.ndim == 3:
        img = img.unsqueeze(0)
    img = img[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(img)


def _download_image(url: str) -> Image.Image:
    req = Request(url, headers={"User-Agent": "flux2-ui"})
    with urlopen(req, timeout=10) as resp:
        if resp.status != 200:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image ({resp.status})")
        data = resp.read()
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid image URL") from exc


app = FastAPI()


@app.on_event("startup")
def _startup() -> None:
    _load_temp_index()


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/models")
def list_models() -> JSONResponse:
    return JSONResponse({"models": MODELS, "defaults": MODEL_DEFAULTS, "active_model": _ACTIVE_MODEL})


@app.get("/temp/list")
def temp_list() -> JSONResponse:
    items = sorted(_TEMP_INDEX.values(), key=lambda x: x["created_at"], reverse=True)
    return JSONResponse({"items": items})


@app.get("/temp/{temp_id}")
def temp_get(temp_id: str) -> FileResponse:
    info = _TEMP_INDEX.get(temp_id)
    if not info:
        raise HTTPException(status_code=404, detail="Unknown temp image")
    path = TEMP_DIR / info["filename"]
    return FileResponse(path)


@app.post("/temp/save")
async def temp_save(file: UploadFile = File(...), label: str | None = Form(default=None)) -> JSONResponse:
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid image file") from exc
    temp_id = _save_temp_image(img, label=label)
    return JSONResponse({"id": temp_id})


@app.post("/temp/delete/{temp_id}")
def temp_delete(temp_id: str) -> JSONResponse:
    info = _TEMP_INDEX.pop(temp_id, None)
    if not info:
        raise HTTPException(status_code=404, detail="Unknown temp image")
    path = TEMP_DIR / info["filename"]
    if path.exists():
        path.unlink()
    _save_temp_index()
    return JSONResponse({"ok": True})


@app.post("/load-url")
def load_url(url: str = Form(...)) -> JSONResponse:
    try:
        img = _download_image(url)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    temp_id = _save_temp_image(img, label="url")
    return JSONResponse({"id": temp_id})


@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    neg_prompt: str = Form(default=""),
    width: int = Form(default=512),
    height: int = Form(default=512),
    model: str = Form(default="klein-4B"),
    num_steps: int = Form(default=4),
    guidance: float | None = Form(default=None),
    cfg_scale: float = Form(default=1.0),
    seed: int | None = Form(default=None),
    temp_input_ids: str = Form(default="[]"),
    save_temp: bool = Form(default=False),
    files: list[UploadFile] | None = File(default=None),
) -> Response:
    width, height = _validate_img_size(width, height)
    pipeline = _get_pipeline()
    if model != _ACTIVE_MODEL:
        raise HTTPException(status_code=400, detail="Selected model is not loaded")

    ref_imgs: list[Image.Image] = []

    try:
        temp_ids = json.loads(temp_input_ids)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid temp_input_ids JSON") from exc

    for temp_id in temp_ids:
        ref_imgs.append(_get_temp_image(temp_id).convert("RGB"))

    if files:
        for upload in files:
            data = await upload.read()
            try:
                img = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=400, detail="Invalid uploaded image") from exc
            ref_imgs.append(img)

    if not ref_imgs:
        ref_imgs = None

    guidance_value = guidance
    if guidance_value is not None and np.isnan(guidance_value):
        guidance_value = None

    output = pipeline.generate(
        prompt=prompt,
        neg_prompt=neg_prompt,
        ref_imgs=ref_imgs,
        img_size=(height, width),
        guidance=guidance_value,
        cfg_scale=cfg_scale,
        num_steps=num_steps,
        seed=seed,
        pbar=False,
    )

    pil_img = _tensor_to_pil(output)

    temp_id = None
    if save_temp:
        temp_id = _save_temp_image(pil_img, label="generated")

    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    headers = {
        "X-Width": str(width),
        "X-Height": str(height),
    }
    if temp_id:
        headers["X-Temp-Id"] = temp_id

    return Response(content=img_bytes.getvalue(), media_type="image/png", headers=headers)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.post("/model/load")
def load_model(model: str = Form(...)) -> JSONResponse:
    pipeline = _load_model(model)
    return JSONResponse({"ok": True, "model": _ACTIVE_MODEL, "device": str(next(pipeline.flux.parameters()).device)})


@app.get("/vram")
def vram() -> JSONResponse:
    if not torch.cuda.is_available():
        raise HTTPException(status_code=500, detail="CUDA is required for Flux.2 UI")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    used_bytes = total_bytes - free_bytes
    return JSONResponse(
        {
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "used_bytes": used_bytes,
        }
    )
