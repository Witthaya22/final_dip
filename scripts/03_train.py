# -*- coding: utf-8 -*-
"""
=====================================================================
 STEP 3 : เทรน YOLO 2 แบบ  (Rectangle  และ  Polygon)
=====================================================================

    แบบ Rectangle  -> โมเดล yolo26n.pt      + rect.yaml   (task = detect)
                       ผลลัพธ์ = กรอบสี่เหลี่ยมรอบวัตถุ
    แบบ Polygon    -> โมเดล yolo26n-seg.pt  + poly.yaml   (task = segment)
                       ผลลัพธ์ = กรอบ + เส้นรอบรูปวัตถุ (mask)

แนวคิด Transfer Learning:
    เราไม่ได้เริ่มเทรนจากศูนย์ แต่โหลด weight ที่ฝึกมาแล้ว (pretrained บน COCO)
    แล้ว "ฝึกต่อ" ให้รู้จักแค่ 4 คลาสของเรา -> ใช้ภาพน้อย เทรนเร็ว แม่นกว่า

ผลลัพธ์ที่ได้ (ไว้ส่งอาจารย์):
    runs/rect/  และ  runs/poly/
        weights/best.pt        <- weight ที่ดีที่สุด  (ส่ง Weight)
        results.png / .csv     <- กราฟ loss, mAP ระหว่างเทรน (ส่งผลการเทรน)
        confusion_matrix.png   <- ตารางทายถูก/ผิดแต่ละคลาส
        *_curve.png            <- กราฟ Precision / Recall / F1
    weights/rect_best.pt, weights/poly_best.pt  <- copy ไว้ให้เว็บใช้

วิธีใช้ (รันจากโฟลเดอร์ final_dip/):
    python scripts/03_train.py --mode both --epochs 20
    python scripts/03_train.py --mode rect           # เทรนเฉพาะแบบกรอบ
    python scripts/03_train.py --mode poly           # เทรนเฉพาะแบบ polygon
"""

import argparse
import os
import shutil
from pathlib import Path

# ป้องกัน error "OMP: Error #15" บน Windows (เหมือนใน lab)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO  # noqa: E402  (import หลังตั้งค่า env)

ROOT = Path(__file__).resolve().parent.parent  # = final_dip/

# ตั้งค่าของแต่ละแบบ: (weight เริ่มต้น, ไฟล์ yaml)
CONFIGS = {
    "rect": ("yolo26n.pt", "rect.yaml"),
    "poly": ("yolo26n-seg.pt", "poly.yaml"),
}


def train_one(mode, epochs, imgsz, batch, device):
    pretrained, data_yaml = CONFIGS[mode]
    print(f"\n========== TRAIN {mode.upper()}  ({pretrained} + {data_yaml}) ==========")

    # 1) โหลดโมเดล pretrained  (ถ้าไม่มีไฟล์ ultralytics จะดาวน์โหลดให้เอง)
    model = YOLO(str(ROOT / pretrained) if (ROOT / pretrained).exists() else pretrained)

    # 2) เทรน
    model.train(
        data=data_yaml,     # ไฟล์บอกว่า dataset อยู่ไหน มีกี่คลาส
        epochs=epochs,      # จำนวนรอบที่ดูข้อมูลทั้งหมด
        imgsz=imgsz,        # ย่อ/ขยายภาพเป็นขนาดนี้ก่อนเข้าโมเดล
        batch=batch,        # จำนวนภาพต่อ 1 step
        patience=15,        # ถ้า 15 epoch ติดกันไม่ดีขึ้น -> หยุดก่อน (early stopping)
        device=device,      # "cpu" หรือ 0 (GPU)
        workers=2,
        project=str(ROOT / "runs"),
        name=mode,          # -> runs/rect หรือ runs/poly
        exist_ok=True,      # เขียนทับโฟลเดอร์เดิมได้
        seed=42,
        plots=True,         # วาดกราฟผลการเทรน
    )

    # 3) ทดสอบกับชุด test (ภาพที่โมเดลไม่เคยเห็นตอนเทรน)
    best = ROOT / "runs" / mode / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(data=data_yaml, split="test", imgsz=imgsz, device=device,
                                  project=str(ROOT / "runs"), name=f"{mode}_test", exist_ok=True)
    print(f"[{mode}] TEST box mAP50 = {metrics.box.map50:.3f}   mAP50-95 = {metrics.box.map:.3f}")
    if mode == "poly":
        print(f"[{mode}] TEST mask mAP50 = {metrics.seg.map50:.3f}   mAP50-95 = {metrics.seg.map:.3f}")

    # 4) copy best.pt ไปไว้ที่ weights/ ให้เว็บ (backend) โหลดไปใช้
    (ROOT / "weights").mkdir(exist_ok=True)
    shutil.copy2(best, ROOT / "weights" / f"{mode}_best.pt")
    print(f"[{mode}] saved -> weights/{mode}_best.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["rect", "poly", "both"], default="both")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cpu", help="cpu หรือ 0 ถ้ามี GPU")
    args = p.parse_args()

    os.chdir(ROOT)  # ให้ path ใน yaml (./datasets/...) อ้างอิงจาก final_dip/ เสมอ
    modes = ["rect", "poly"] if args.mode == "both" else [args.mode]
    for m in modes:
        train_one(m, args.epochs, args.imgsz, args.batch, args.device)


if __name__ == "__main__":  # จำเป็นบน Windows เพราะ dataloader ใช้ multiprocessing
    main()
