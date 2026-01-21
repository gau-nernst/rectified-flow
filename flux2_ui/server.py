import gc
import io
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

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

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
def health():
    return None


@app.get("/")
def index():
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


@app.get("/models")
def list_models():
    return {"models": MODELS, "defaults": MODEL_DEFAULTS, "active_model": ACTIVE_MODEL}


@app.get("/image", response_class=JSONResponse)
def image_list():
    items = []
    for path in IMAGE_DIR.iterdir():
        if not (path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}):
            continue
        items.append({"filename": path.name})
    return items


@app.get("/image/{filename}", response_class=FileResponse)
def image_get(filename: str):
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(status_code=404, detail="Unknown image")
    path = IMAGE_DIR / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Unknown image")
    return FileResponse(path)


@app.post("/image", response_class=JSONResponse)
async def image_import(
    file: UploadFile = File(...),
):
    data = await file.read()
    filename = file.filename or ""
    if not filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if "/" in filename or "\\" in filename or Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = IMAGE_DIR / filename
    if path.exists():
        raise HTTPException(status_code=409, detail="File already exists")
    Image.open(io.BytesIO(data))
    path.write_bytes(data)


@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    neg_prompt: str = Form(""),
    width: int = Form(512),
    height: int = Form(512),
    num_steps: int = Form(4),
    cfg_scale: float = Form(1.0),
    seed: int | None = Form(None),
    images: list[UploadFile] = File(default=[]),
):
    ref_imgs = []
    for image in images:
        data = await image.read()
        ref_imgs.append(Image.open(io.BytesIO(data)).convert("RGB"))
    if not ref_imgs:
        ref_imgs = None

    img_pt = PIPELINE.generate(
        prompt=prompt,
        neg_prompt=neg_prompt,
        ref_imgs=ref_imgs,
        img_size=(height, width),
        cfg_scale=cfg_scale,
        num_steps=num_steps,
        seed=seed,
        pbar=False,
    )
    img_np = img_pt.squeeze(0).permute(1, 2, 0).cpu().numpy()
    pil_img = Image.fromarray(img_np)

    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format="WEBP", lossless=True)
    img_bytes.seek(0)

    return Response(content=img_bytes.getvalue(), media_type="image/webp")


@app.post("/model/{model_name}")
def load_model(model_name: str):
    global PIPELINE, ACTIVE_MODEL
    if PIPELINE is not None:
        del PIPELINE
        gc.collect()
        torch.cuda.empty_cache()

    PIPELINE = Flux2Pipeline(flux=load_flux2(model_name)).cuda()
    ACTIVE_MODEL = model_name


@app.get("/vram")
def vram() -> JSONResponse:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    used_bytes = total_bytes - free_bytes
    return {
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "used_bytes": used_bytes,
    }
