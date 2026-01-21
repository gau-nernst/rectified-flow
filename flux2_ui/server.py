import gc
import io
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import requests
import torch
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from modelling import Flux2Pipeline, load_flux2

APP_ROOT = Path(__file__).resolve().parent
STATIC_DIR = APP_ROOT / "static"
IMAGE_DIR = APP_ROOT / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["klein-4B", "klein-9B", "klein-base-4B", "klein-base-9B"]
MODEL_DEFAULTS = {
    "klein-4B": dict(cfg_scale=1.0, num_steps=4),
    "klein-9B": dict(cfg_scale=1.0, num_steps=4),
    "klein-base-4B": dict(cfg_scale=4.0, num_steps=50),
    "klein-base-9B": dict(cfg_scale=4.0, num_steps=50),
}

PIPELINE: Flux2Pipeline | None = None
ACTIVE_MODEL: str | None = None

COUNTER = {"flux": 0, "import": 0}


def _safe_filename(name: str) -> str:
    return Path(name).name


image_router = APIRouter(tags=["image"])


@image_router.get("/", response_class=JSONResponse)
@image_router.get("", response_class=JSONResponse, include_in_schema=False)
def image_list():
    items = []

    for path in IMAGE_DIR.iterdir():
        if not (path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}):
            continue

        filename = path.name
        with Image.open(path) as img:
            width, height = img.size

        entry = {
            "filename": filename,
            "created_at": path.stat().st_mtime,
            "width": width,
            "height": height,
        }
        items.append(entry)

    items.sort(key=lambda item: item["created_at"], reverse=True)
    return {"items": items}


@image_router.post("/", response_class=JSONResponse)
async def image_import(
    file: UploadFile | None = File(default=None),
    url: str | None = Form(default=None),
    is_flux: bool = Form(default=False),
):
    if file is not None:
        data = await file.read()

    elif url:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.content

    else:
        raise HTTPException(status_code=400, detail="Provide a file or url")

    fmt = Image.open(io.BytesIO(data)).format
    key = "flux" if is_flux else "import"

    filename = f"{key}_{COUNTER[key]}.{fmt}"
    (IMAGE_DIR / filename).write_bytes(data)
    COUNTER[key] += 1

    return {"filename": filename}


@image_router.get("/{filename}", response_class=FileResponse)
def image_get(filename: str):
    path = IMAGE_DIR / _safe_filename(filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Unknown image")
    return FileResponse(path)


@image_router.delete("/{filename}")
def image_delete(filename: str):
    path = IMAGE_DIR / _safe_filename(filename)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Unknown image")
    path.unlink()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # init COUNTER
    for path in IMAGE_DIR.iterdir():
        if not (path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}):
            continue

        # file is named like flux_0012.webp
        # attempt to extract key and index. if failed, move on.
        name = path.stem
        try:
            key, idx = name.split("_")
            COUNTER[key] = max(COUNTER[key], int(idx))
        except Exception:
            continue

    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(image_router, prefix="/image")


@app.get("/health")
def health():
    return None


@app.get("/")
def index():
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


@app.get("/models")
def list_models():
    return {"models": MODELS, "defaults": MODEL_DEFAULTS, "active_model": ACTIVE_MODEL}


class GenerateRequest(BaseModel):
    prompt: str
    neg_prompt: str = ""
    width: int = 512
    height: int = 512
    num_steps: int = 4
    cfg_scale: float = 1.0
    seed: int | None = None
    image_input_filenames: list[str] = Field(default_factory=list)


@lru_cache(maxsize=10)
def load_img_rgb(filename: str) -> Image.Image:
    path = IMAGE_DIR / _safe_filename(filename)
    return Image.open(path).convert("RGB")


@app.post("/generate")
async def generate(req: GenerateRequest):
    if req.image_input_filenames:
        ref_imgs = [load_img_rgb(filename) for filename in req.image_input_filenames]
    else:
        ref_imgs = None

    img_pt = PIPELINE.generate(
        prompt=req.prompt,
        neg_prompt=req.neg_prompt,
        ref_imgs=ref_imgs,
        img_size=(req.height, req.width),
        cfg_scale=req.cfg_scale,
        num_steps=req.num_steps,
        seed=req.seed,
        pbar=False,
    )
    img_np = img_pt.squeeze(0).permute(1, 2, 0).cpu().numpy()
    pil_img = Image.fromarray(img_np)

    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format="WEBP", lossless=True)
    img_bytes.seek(0)

    return Response(content=img_bytes.getvalue(), media_type="image/webp")


@app.post("/model/load")
def load_model(model: str):
    global PIPELINE, ACTIVE_MODEL
    if PIPELINE is not None:
        del PIPELINE
        gc.collect()
        torch.cuda.empty_cache()

    PIPELINE = Flux2Pipeline(flux=load_flux2(model)).cuda()
    ACTIVE_MODEL = model


@app.get("/vram")
def vram() -> JSONResponse:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    used_bytes = total_bytes - free_bytes
    return {
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "used_bytes": used_bytes,
    }
