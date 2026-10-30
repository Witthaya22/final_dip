# -*- coding: utf-8 -*-
"""
=====================================================================
 STEP 5 : Backend (หลังบ้าน) - REST API ด้วย FastAPI
=====================================================================

หน้าที่ของหลังบ้าน:
    - โหลด weight โมเดล Polygon ที่เทรนเสร็จ (weights/poly_best.pt) ไว้ในหน่วยความจำ
    - รับภาพที่ผู้ใช้อัปโหลดจากหน้าเว็บ  ->  ส่งให้ YOLO ทำนาย
    - ส่งผลกลับเป็น JSON (ชื่อคลาส, ความมั่นใจ, พิกัดกรอบ, พิกัด polygon)
    - เสิร์ฟไฟล์หน้าเว็บ (frontend) ด้วย  -> deploy ที่เดียวใช้ได้ทั้งหน้าบ้านและหลังบ้าน

API ที่มี:
    GET  /api/health    -> เช็คว่า server พร้อม + รายชื่อคลาส
    POST /api/predict   -> ส่งภาพมาทำนาย  (form-data: file, conf=0.25)
    GET  /docs          -> หน้าเอกสาร API อัตโนมัติของ FastAPI (ลองยิง API ได้เลย)

วิธีรัน (จากโฟลเดอร์ final_dip/):
    uvicorn web.backend.app:app --reload --port 8000
    แล้วเปิด http://localhost:8000
"""

import io
import os
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np                                     # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware     # noqa: E402
from fastapi.responses import FileResponse             # noqa: E402
from fastapi.staticfiles import StaticFiles            # noqa: E402
from PIL import Image, ImageOps                        # noqa: E402
from ultralytics import YOLO                           # noqa: E402

# ---------------------------------------------------------------
# หา path ต่าง ๆ  (final_dip/web/backend/app.py -> ROOT = final_dip/)
# ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", ROOT / "weights"))
FRONTEND_DIR = ROOT / "web" / "frontend"
MAX_UPLOAD_MB = 10

# ---------------------------------------------------------------
# โหลดโมเดลครั้งเดียวตอนเปิด server (ถ้าโหลดทุก request จะช้ามาก)
# เว็บใช้โมเดล Polygon (segment) อย่างเดียว
# ---------------------------------------------------------------
WEIGHT = WEIGHTS_DIR / "poly_best.pt"
model = YOLO(str(WEIGHT)) if WEIGHT.exists() else None
print(f"[backend] {'loaded' if model else 'WARNING: not found'} {WEIGHT}")

# ---------------------------------------------------------------
# สร้างแอป FastAPI
# ---------------------------------------------------------------
app = FastAPI(title="final_dip YOLO API",
              description="Upload an image -> YOLO (Polygon / segment) predicts airplane / boat / car / motorcycle")

# CORS = อนุญาตให้หน้าเว็บที่อยู่คนละโดเมน (เช่น GitHub Pages) เรียก API นี้ได้
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    """เช็คสถานะ server: โหลดโมเดลได้ไหม และมีคลาสอะไร"""
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
    """รับภาพ 1 ภาพ -> ส่งให้ YOLO (Polygon) -> คืนผลเป็น JSON"""
    # ---- 1) ตรวจสอบ input ----
    if model is None:
        raise HTTPException(503, "model not loaded (ไม่พบ weights/poly_best.pt)")
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"file too large (max {MAX_UPLOAD_MB} MB)")
    try:
        # exif_transpose = หมุนภาพตามข้อมูลกล้อง (ภาพจากมือถือมักตะแคง)
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except Exception:
        raise HTTPException(400, "cannot read image file")

    # ---- 2) ส่งภาพให้ YOLO ทำนาย ----
    # ultralytics รับ numpy array แบบ BGR (แบบ OpenCV) -> กลับลำดับสีจาก RGB
    bgr = np.array(img)[:, :, ::-1]
    t0 = time.perf_counter()
    result = model.predict(bgr, imgsz=512, conf=conf, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000

    # ---- 3) แปลงผลลัพธ์เป็น JSON ----
    names = result.names                      # {0: 'airplane', 1: 'boat', ...}
    boxes = result.boxes                      # กรอบทั้งหมดที่เจอ
    polys = result.masks.xy if result.masks is not None else None  # จุด polygon ของแต่ละวัตถุ

    detections = []
    for i in range(len(boxes)):
        det = {
            "class_id": int(boxes.cls[i]),
            "class_name": names[int(boxes.cls[i])],
            "confidence": round(float(boxes.conf[i]), 4),
            "box": [round(v, 1) for v in boxes.xyxy[i].tolist()],   # [x1, y1, x2, y2] หน่วย pixel
        }
        if polys is not None:
            det["polygon"] = [[round(float(x), 1), round(float(y), 1)] for x, y in polys[i]]
        detections.append(det)

    # เรียงจากมั่นใจมากไปน้อย -> ตัวแรก = คำตอบหลักของภาพ
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


# ---------------------------------------------------------------
# เสิร์ฟหน้าเว็บ (frontend) : เปิด "/" แล้วได้ index.html
# ต้องประกาศไว้ "หลัง" route /api เสมอ ไม่งั้นจะทับกัน
# ---------------------------------------------------------------
if FRONTEND_DIR.exists():
    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
