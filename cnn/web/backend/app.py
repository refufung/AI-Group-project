"""FastAPI inference server for the dog-breed classifier (ResNetSE champion, SmallCNN legacy)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms

from model import build_model

CNN_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = CNN_ROOT / "outputs"
# Prefer the new from-scratch ResNetSE champion; fall back to the legacy SmallCNN.
CKPT_CANDIDATES = [OUTPUTS / "best_resnet_se.pt", OUTPUTS / "best_small_cnn.pt"]
META_PATH = OUTPUTS / "split_meta.json"
FRONTEND_DIST = CNN_ROOT / "web" / "frontend" / "dist"

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Map folder-style labels from batch uploads to class names
FOLDER_TO_CLASS = {
    "french_bulldog": "French_bulldog",
    "frenchbulldog": "French_bulldog",
    "cardigan": "Cardigan",
    "labrador_retriever": "Labrador_retriever",
    "labrador": "Labrador_retriever",
    "golden_retriever": "golden_retriever",
    "goldenretriever": "golden_retriever",
}


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def infer_true_label(relative_path: str | None, class_names: list[str]) -> str | None:
    """If client sends path like Cardigan/n123.jpg, treat parent folder as label."""
    if not relative_path:
        return None
    parts = Path(relative_path.replace("\\", "/")).parts
    if len(parts) < 2:
        return None
    folder = parts[-2]
    key = folder.lower().replace(" ", "_").replace("-", "_")
    mapped = FOLDER_TO_CLASS.get(key)
    if mapped and mapped in class_names:
        return mapped
    # exact match ignoring case
    for name in class_names:
        if name.lower() == folder.lower():
            return name
    return None


class Predictor:
    def __init__(self) -> None:
        self.ckpt_path = next((p for p in CKPT_CANDIDATES if p.is_file()), None)
        if self.ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint found in: {CKPT_CANDIDATES}")

        meta: dict = {}
        if META_PATH.is_file():
            with META_PATH.open(encoding="utf-8") as f:
                meta = json.load(f)

        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.class_names: list[str] = list(
            ckpt.get("class_names")
            or ["French_bulldog", "Cardigan", "Labrador_retriever", "golden_retriever"]
        )
        arch = str(ckpt.get("arch", "small_cnn"))
        cfg = ckpt.get("config") or {}
        dropout = float(ckpt.get("dropout", cfg.get("dropout", 0.3)))
        self.device = get_device()
        self.model = build_model(arch, num_classes=len(self.class_names), dropout=dropout)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Normalization stats live inside new checkpoints; fall back to split_meta.json.
        mean = ckpt.get("mean") or meta["mean"]
        std = ckpt.get("std") or meta["std"]
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        self.meta = {
            "seed": meta.get("seed"),
            "mean": mean,
            "std": std,
            "lr": ckpt.get("lr"),
            "dropout": dropout,
            "best_val_macro_f1": ckpt.get("best_val_macro_f1"),
        }

    def predict_pil(self, image: Image.Image) -> dict:
        rgb = image.convert("RGB")
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu()
        pred_idx = int(probs.argmax().item())
        probs_list = [
            {"class_name": name, "probability": float(probs[i].item())}
            for i, name in enumerate(self.class_names)
        ]
        probs_list.sort(key=lambda x: -x["probability"])
        return {
            "predicted_class": self.class_names[pred_idx],
            "confidence": float(probs[pred_idx].item()),
            "probabilities": probs_list,
        }


app = FastAPI(title="Dog Breed CNN API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor: Predictor | None = None


@app.on_event("startup")  # fine for local demo
def _startup() -> None:
    global predictor
    predictor = Predictor()
    print(f"Model loaded on {predictor.device}")
    print(f"Classes: {predictor.class_names}")


@app.get("/health")
def health() -> dict:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "device": str(predictor.device),
        "class_names": predictor.class_names,
        "checkpoint": str(predictor.ckpt_path),
        "meta": predictor.meta,
    }


async def _read_image(file: UploadFile) -> Image.Image:
    name = file.filename or "upload.jpg"
    suffix = Path(name).suffix.lower()
    if suffix and suffix not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix}. Use: {sorted(ALLOWED_EXT)}",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"Empty file: {name}")
    try:
        return Image.open(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Cannot open image {name}: {exc}") from exc


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    image = await _read_image(file)
    result = predictor.predict_pil(image)
    result["filename"] = file.filename
    return result


@app.post("/predict/batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    relative_paths: list[str] | None = Form(None),
) -> dict:
    """Batch predict. Optional relative_paths[i] like 'Cardigan/n123.jpg' for true labels."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Form may send a single string or repeated fields; normalize to list
    paths: list[str | None]
    if relative_paths is None:
        paths = [None] * len(files)
    elif isinstance(relative_paths, str):
        paths = [relative_paths] + [None] * (len(files) - 1)
    else:
        paths = list(relative_paths)
        if len(paths) < len(files):
            paths.extend([None] * (len(files) - len(paths)))

    results = []
    labeled = 0
    correct = 0
    for file, rel in zip(files, paths):
        name = file.filename or "unknown"
        suffix = Path(name).suffix.lower()
        if suffix and suffix not in ALLOWED_EXT:
            results.append(
                {
                    "filename": name,
                    "relative_path": rel,
                    "error": f"Unsupported type {suffix}",
                }
            )
            continue
        try:
            image = await _read_image(file)
            pred = predictor.predict_pil(image)
        except HTTPException as exc:
            results.append(
                {
                    "filename": name,
                    "relative_path": rel,
                    "error": exc.detail,
                }
            )
            continue

        true_label = infer_true_label(rel or name, predictor.class_names)
        item = {
            "filename": name,
            "relative_path": rel,
            "true_label": true_label,
            **pred,
        }
        if true_label is not None:
            labeled += 1
            is_correct = true_label == pred["predicted_class"]
            item["correct"] = is_correct
            if is_correct:
                correct += 1
        results.append(item)

    summary = {
        "n_total": len(results),
        "n_ok": sum(1 for r in results if "error" not in r),
        "n_labeled": labeled,
        "n_correct": correct,
        "accuracy": (correct / labeled) if labeled else None,
    }
    return {"summary": summary, "results": results, "class_names": predictor.class_names}


# Serve the built React app (same origin for tunnel / Docker deploys)
if FRONTEND_DIST.is_dir():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/")
    def spa_index() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str) -> FileResponse:
        # Do not shadow API routes (already registered above)
        candidate = FRONTEND_DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")

