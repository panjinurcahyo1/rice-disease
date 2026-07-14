"""
Rice Leaf Disease Classifier — FastAPI backend

Default listen port is 8080 (8000 is often busy). Override with env APP_PORT.

Run (CPU inference by default):

    cd web_app
    uvicorn app:app --host 0.0.0.0 --port 8080

Or:

    python app.py

--host 0.0.0.0 means listen on all interfaces (LAN / deploy). Browsers cannot open
http://0.0.0.0:PORT/ (ERR_ADDRESS_INVALID). Always use http://127.0.0.1:PORT/ or
http://localhost:PORT/ — same machine, same server, valid URL.

GPU (optional): set USE_GPU=1 before starting uvicorn.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

# --- Must run before `import torch` -----------------------------------------
# Conda + MKL on Windows often loads two OpenMP runtimes → OMP #15 crash/warning.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Web deploy: CPU-only inference by default (no VRAM, fewer driver issues).
# Set USE_GPU=1 to allow CUDA if you really want the GPU.
_USE_GPU = os.environ.get("USE_GPU", "0").lower() in ("1", "true", "yes")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import models, transforms  # noqa: E402

# ---------------------------------------------------------------------------
# Locate checkpoint
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent          # one level above web_app/
RESULTS_ROOT = REPO_ROOT / "results"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if _USE_GPU and torch.cuda.is_available() else "cpu")


def _find_best_checkpoint() -> Path:
    """
    Walk every run folder under results/ and pick the best_model.pt that
    belongs to the run with the highest val_acc (stored inside the checkpoint).
    Falls back to the most-recently-modified file if none embed val_acc.
    """
    candidates: list[tuple[float, Path]] = []
    for p in sorted(RESULTS_ROOT.glob("*/best_model.pt")):
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            val_acc = float(ckpt.get("val_acc", 0.0))
            candidates.append((val_acc, p))
        except Exception:
            candidates.append((0.0, p))
    if not candidates:
        raise FileNotFoundError(
            f"No best_model.pt found under {RESULTS_ROOT}. "
            "Train the model first (run notebook.ipynb)."
        )
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


_ckpt_env = os.environ.get("CKPT_PATH", "")
CKPT_PATH = Path(_ckpt_env) if _ckpt_env else _find_best_checkpoint()

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def _build_resnet18(num_classes: int, dropout: float = 0.0) -> nn.Module:
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(m.fc.in_features, num_classes),
    )
    return m


def _build_resnet34(num_classes: int, dropout: float = 0.0) -> nn.Module:
    m = models.resnet34(weights=None)
    m.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(m.fc.in_features, num_classes),
    )
    return m


_BUILDERS = {
    "resnet18": _build_resnet18,
    "resnet34": _build_resnet34,
}


def load_model(ckpt_path: Path):
    ckpt     = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    classes  = ckpt["classes"]
    img_size = ckpt.get("img_size", 224)
    cfg      = ckpt.get("config", {})
    dropout  = cfg.get("dropout", 0.0)
    arch     = cfg.get("model_name", "resnet18")

    builder = _BUILDERS.get(arch, _build_resnet18)
    m = builder(len(classes), dropout=dropout).to(DEVICE)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m, classes, img_size


model, CLASS_NAMES, IMG_SIZE = load_model(CKPT_PATH)
print(f"Loaded  : {CKPT_PATH}")
print(f"Device  : {DEVICE}  (USE_GPU={_USE_GPU}, torch.cuda.is_available()={torch.cuda.is_available()})")
print(f"Classes : {CLASS_NAMES}")
print(f"Img size: {IMG_SIZE}")


def _make_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


TFM = _make_transform(IMG_SIZE)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Rice Leaf Disease Classifier", version="1.0.0")

# Serve static files (HTML/CSS/JS) from web_app/static/
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def _print_browser_url_hint() -> None:
    """Uvicorn logs '0.0.0.0:PORT' which misleads people into pasting that into the browser."""
    port = os.environ.get("APP_PORT", "").strip()
    port_line = (
        f"    http://127.0.0.1:{port}/\n    http://localhost:{port}/\n"
        if port.isdigit()
        else "    http://127.0.0.1:<PORT>/\n    http://localhost:<PORT>/\n"
        "    (replace <PORT> with the number in Uvicorn's startup line, e.g. 8080)\n"
    )
    print(
        "\n"
        + "=" * 66
        + "\n  In your browser, open ONE of:\n"
        + port_line
        + "  Never http://0.0.0.0:... — invalid in Chrome / Edge / Firefox.\n"
        + "=" * 66
        + "\n",
        flush=True,
    )


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "use_gpu_requested": _USE_GPU,
        "cuda_available": torch.cuda.is_available(),
        "classes": CLASS_NAMES,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate MIME type loosely
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {exc}") from exc

    x = TFM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    ranked = sorted(
        [{"class": CLASS_NAMES[i], "probability": float(probs[i])} for i in range(len(CLASS_NAMES))],
        key=lambda d: d["probability"],
        reverse=True,
    )

    return JSONResponse({
        "top_class":   ranked[0]["class"],
        "top_prob":    ranked[0]["probability"],
        "predictions": ranked,
    })


if __name__ == "__main__":
    import uvicorn

    _port = int(os.environ.get("APP_PORT", "8080"))
    os.environ["APP_PORT"] = str(_port)  # so startup banner matches this run
    uvicorn.run("app:app", host="0.0.0.0", port=_port, reload=False)
