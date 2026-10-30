# -*- coding: utf-8 -*-
"""
=====================================================================
 STEP 6 : แปลง weight (.pt) -> ONNX (.onnx) เพื่อให้รันในเบราว์เซอร์ได้
=====================================================================

ทำไมต้องแปลง?
    hosting ฟรีส่วนใหญ่ (Hugging Face Static Space, GitHub Pages) รันได้แค่ไฟล์เว็บ
    ไม่มี Python ให้รัน PyTorch  ->  เราแปลงโมเดลเป็น ONNX ซึ่งเป็นรูปแบบกลาง
    แล้วใช้ "ONNX Runtime Web" (JavaScript) รันโมเดลในเบราว์เซอร์ของผู้ใช้แทน
    ดูโค้ดฝั่งเว็บได้ที่ web/frontend/infer.js

ผลลัพธ์:
    web/frontend/models/rect.onnx   (โมเดล Rectangle)
    web/frontend/models/poly.onnx   (โมเดล Polygon)

วิธีใช้ (รันจากโฟลเดอร์ final_dip/):
    python scripts/06_export_onnx.py
"""

import os
import shutil
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "frontend" / "models"
IMGSZ = 512  # ต้องตรงกับ IMGSZ ใน infer.js


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for mode in ("rect", "poly"):
        weight = ROOT / "weights" / f"{mode}_best.pt"
        if not weight.exists():
            print(f"[{mode}] ไม่พบ {weight} -> รัน 03_train.py ก่อน")
            continue
        # export จะสร้างไฟล์ .onnx ไว้ข้าง ๆ ไฟล์ .pt
        #   simplify=True -> ลดโหนดที่ไม่จำเป็น ไฟล์เล็กลง/เร็วขึ้น
        onnx_path = YOLO(str(weight)).export(format="onnx", imgsz=IMGSZ, simplify=True)
        shutil.move(onnx_path, OUT / f"{mode}.onnx")
        size = (OUT / f"{mode}.onnx").stat().st_size / 1e6
        print(f"[{mode}] -> web/frontend/models/{mode}.onnx  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
