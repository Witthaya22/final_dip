import io
import os
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", ROOT / "weights"))
FRONTEND_DIR = ROOT / "web" / "frontend"
MAX_UPLOAD_MB = 10

WEIGHT = WEIGHTS_DIR / "poly_best.pt"
model = YOLO(str(WEIGHT)) if WEIGHT.exists() else None
print(f"[backend] {'loaded' if model else 'WARNING: not found'} {WEIGHT}")

app = FastAPI(title="final_dip YOLO API",
              description="Upload an image -> YOLO (Polygon / segment) predicts airplane / boat / car / motorcycle")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models": ["poly"] if model else [],
        "classes": list(model.names.values()) if model else [],
    }


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(..., description="ไฟล์ภาพ jpg/png"),
    conf: float = Form(0.25, description="ความมั่นใจขั้นต่ำ 0-1"),
):
    if model is None:
        raise HTTPException(503, "model not loaded (ไม่พบ weights/poly_best.pt)")
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"file too large (max {MAX_UPLOAD_MB} MB)")
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except Exception:
        raise HTTPException(400, "cannot read image file")

    bgr = np.array(img)[:, :, ::-1]
    t0 = time.perf_counter()
    result = model.predict(bgr, imgsz=512, conf=conf, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000

    names = result.names
    boxes = result.boxes
    polys = result.masks.xy if result.masks is not None else None

    detections = []
    for i in range(len(boxes)):
        det = {
            "class_id": int(boxes.cls[i]),
            "class_name": names[int(boxes.cls[i])],
            "confidence": round(float(boxes.conf[i]), 4),
            "box": [round(v, 1) for v in boxes.xyxy[i].tolist()],
        }
        if polys is not None:
            det["polygon"] = [[round(float(x), 1), round(float(y), 1)] for x, y in polys[i]]
        detections.append(det)

    detections.sort(key=lambda d: d["confidence"], reverse=True)

    return {
        "mode": "poly",
        "image_width": img.width,
        "image_height": img.height,
        "inference_ms": round(ms, 1),
        "count": len(detections),
        "top_class": detections[0]["class_name"] if detections else None,
        "detections": detections,
    }


if FRONTEND_DIR.exists():
    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
