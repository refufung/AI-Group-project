"""FastAPI inference server for the trained EfficientNet-B0 dog-breed classifier."""

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

from model import build_efficientnet_b0

ENET_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ENET_ROOT / "outputs"
CKPT_PATH = OUTPUTS / "best_efficientnet_b0.pt"
META_PATH = OUTPUTS / "split_meta.json"
METRICS_PATH = OUTPUTS / "metrics.json"
FRONTEND_DIST = ENET_ROOT / "web" / "frontend" / "dist"

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

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
    for name in class_names:
        if name.lower() == folder.lower():
            return name
    return None


class Predictor:
    def __init__(self) -> None:
        if not CKPT_PATH.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

        meta: dict = {}
        if META_PATH.is_file():
            with META_PATH.open(encoding="utf-8") as f:
                meta = json.load(f)

        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        self.class_names: list[str] = list(
            ckpt.get("class_names")
            or ["French_bulldog", "Cardigan", "Labrador_retriever", "golden_retriever"]
        )
        self.device = get_device()
        width_mult = float(ckpt.get("width_mult", 0.5))
        dropout = float(ckpt.get("dropout", 0.2))
        depth_drop = int(ckpt.get("depth_drop", 0))
        depth_mult = float(ckpt.get("depth_mult", 1.0))
        se_ratio = float(ckpt.get("se_ratio", 0.25))
        drop_path_rate = float(ckpt.get("drop_path_rate", 0.0))
        image_size = int(ckpt.get("image_size", meta.get("image_size", 224)))
        self.model = build_efficientnet_b0(
            num_classes=len(self.class_names),
            pretrained=False,
            width_mult=width_mult,
            dropout=dropout,
            depth_drop=depth_drop,
            depth_mult=depth_mult,
            se_ratio=se_ratio,
            drop_path_rate=drop_path_rate,
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        mean = ckpt.get("mean") or meta.get("mean") or [0.485, 0.456, 0.406]
        std = ckpt.get("std") or meta.get("std") or [0.229, 0.224, 0.225]
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        test_acc = None
        if METRICS_PATH.is_file():
            with METRICS_PATH.open(encoding="utf-8") as f:
                metrics = json.load(f)
            test_acc = (metrics.get("test") or {}).get("accuracy")

        self.meta = {
            "architecture": ckpt.get("architecture", "efficientnet_b0_from_scratch"),
            "seed": ckpt.get("seed", meta.get("seed")),
            "mean": mean,
            "std": std,
            "best_val_macro_f1": ckpt.get("best_val_macro_f1"),
            "best_stage_b_epoch": ckpt.get("best_stage_b_epoch"),
            "test_accuracy": test_acc,
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


app = FastAPI(title="Dog Breed EfficientNet API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor: Predictor | None = None


@app.on_event("startup")
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
        "checkpoint": str(CKPT_PATH),
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
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    if relative_paths is None:
        paths: list[str | None] = [None] * len(files)
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


if FRONTEND_DIST.is_dir():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/")
    def spa_index() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str) -> FileResponse:
        candidate = FRONTEND_DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
