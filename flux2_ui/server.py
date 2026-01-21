import gc
import io
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import requests
import torch
from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from modelling import Flux2Pipeline, load_flux2

APP_ROOT = Path(__file__).resolve().parent
STATIC_DIR = APP_ROOT / "static"
IMAGE_DIR = APP_ROOT / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = IMAGE_DIR / "index.db"

MODELS = ["klein-4B", "klein-9B", "klein-base-4B", "klein-base-9B"]
MODEL_DEFAULTS = {
    "klein-4B": dict(cfg_scale=1.0, num_steps=4),
    "klein-9B": dict(cfg_scale=1.0, num_steps=4),
    "klein-base-4B": dict(cfg_scale=4.0, num_steps=50),
    "klein-base-9B": dict(cfg_scale=4.0, num_steps=50),
}

PIPELINE: Flux2Pipeline | None = None
ACTIVE_MODEL: str | None = None


def _save_image(db_conn: sqlite3.Connection, img: Image.Image, label: str | None = None) -> str:
    image_id = uuid.uuid4().hex
    filename = f"{image_id}.png"
    path = IMAGE_DIR / filename
    img.save(path, format="PNG")
    db_conn.execute(
        """
        INSERT INTO images (id, filename, label, created_at, width, height)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (image_id, filename, label or "", time.time(), img.width, img.height),
    )
    db_conn.commit()
    return image_id


def get_db(req: Request) -> sqlite3.Connection:
    return req.app.state.db_conn


DBDep = Annotated[sqlite3.Connection, Depends(get_db)]

image_router = APIRouter(tags=["image"])


@image_router.get("/list", response_class=JSONResponse)
def image_list(db_conn: DBDep):
    rows = db_conn.execute("SELECT * FROM images ORDER BY created_at DESC").fetchall()
    items = [dict(row) for row in rows]
    return {"items": items}


@image_router.get("/{image_id}", response_class=FileResponse)
def image_get(image_id: str, db_conn: DBDep):
    row = db_conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown image")
    path = IMAGE_DIR / row["filename"]
    return FileResponse(path)


@image_router.post("/save", response_class=JSONResponse)
async def image_save(db_conn: DBDep, file: UploadFile = File(...), label: str | None = Form(default=None)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    image_id = _save_image(db_conn, img, label=label)
    return {"id": image_id}


@image_router.post("/delete/{image_id}")
def image_delete(image_id: str, db_conn: DBDep) -> JSONResponse:
    row = db_conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown image")
    db_conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
    db_conn.commit()
    info = dict(row)
    path = IMAGE_DIR / info["filename"]
    if path.exists():
        path.unlink()
    return {"ok": True}


@image_router.post("/load-url")
def image_load_url(db_conn: DBDep, url: str = Form(...)) -> JSONResponse:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content))
    img_id = _save_image(db_conn, img, label="url")
    return {"id": img_id}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # init db
    db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    db_conn.row_factory = sqlite3.Row
    db_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            label TEXT NOT NULL,
            created_at REAL NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL
        )
        """
    )
    db_conn.commit()
    app.state.db_conn = db_conn

    yield

    db_conn.close()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(image_router, prefix="/image")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


@app.get("/models")
def list_models() -> JSONResponse:
    return {"models": MODELS, "defaults": MODEL_DEFAULTS, "active_model": ACTIVE_MODEL}


class GenerateRequest(BaseModel):
    prompt: str
    neg_prompt: str = ""
    width: int = 512
    height: int = 512
    num_steps: int = 4
    cfg_scale: float = 1.0
    seed: int | None = None
    image_input_ids: list[str] = Field(default_factory=list)


@app.post("/generate")
async def generate(db_conn: DBDep, payload: GenerateRequest):
    ref_imgs: list[Image.Image] = []

    image_ids = payload.image_input_ids

    for image_id in image_ids:
        row = db_conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        path = IMAGE_DIR / row["filename"]
        ref_imgs.append(Image.open(path).convert("RGB"))

    if not ref_imgs:
        ref_imgs = None

    img = PIPELINE.generate(
        prompt=payload.prompt,
        neg_prompt=payload.neg_prompt,
        ref_imgs=ref_imgs,
        img_size=(payload.height, payload.width),
        cfg_scale=payload.cfg_scale,
        num_steps=payload.num_steps,
        seed=payload.seed,
        pbar=False,
    )

    if img.ndim == 3:
        img = img.unsqueeze(0)
    img = img[0].permute(1, 2, 0).cpu().numpy()
    pil_img = Image.fromarray(img)

    image_id = None

    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    headers = {
        "X-Width": str(payload.width),
        "X-Height": str(payload.height),
    }
    if image_id:
        headers["X-Image-Id"] = image_id

    return Response(content=img_bytes.getvalue(), media_type="image/png", headers=headers)


@app.post("/model/load")
def load_model(model: str = Form(...)) -> JSONResponse:
    global PIPELINE, ACTIVE_MODEL
    if PIPELINE is not None:
        del PIPELINE
        gc.collect()
        torch.cuda.empty_cache()

    PIPELINE = Flux2Pipeline(flux=load_flux2(model)).cuda()
    ACTIVE_MODEL = model

    return {"ok": True}


@app.get("/vram")
def vram() -> JSONResponse:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    used_bytes = total_bytes - free_bytes
    return {
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "used_bytes": used_bytes,
    }
