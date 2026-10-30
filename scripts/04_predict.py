# -*- coding: utf-8 -*-
"""
=====================================================================
 STEP 4 : Predict (ทำนาย) ด้วย weight ที่เทรนเสร็จแล้ว
=====================================================================

    ใช้ภาพในชุด test (ภาพที่โมเดลไม่เคยเห็น) ให้โมเดลทาย แล้วเซฟภาพผลลัพธ์
    ที่วาดกรอบ / polygon พร้อมชื่อคลาสและความมั่นใจ (confidence)

ผลลัพธ์ (ส่งผลการ Predict):
    runs/predict_rect/*.jpg   <- ผลจากโมเดล Rectangle
    runs/predict_poly/*.jpg   <- ผลจากโมเดล Polygon

วิธีใช้ (รันจากโฟลเดอร์ final_dip/):
    python scripts/04_predict.py                     # ใช้ภาพ test ทั้งหมด
    python scripts/04_predict.py --source my.jpg     # ทายภาพของเราเอง
"""

import argparse
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=None, help="ไฟล์ภาพ / โฟลเดอร์ (ค่าเริ่มต้น = ภาพ test)")
    p.add_argument("--conf", type=float, default=0.25, help="ความมั่นใจขั้นต่ำที่จะแสดงผล")
    args = p.parse_args()

    for mode in ("rect", "poly"):
        weight = ROOT / "weights" / f"{mode}_best.pt"
        if not weight.exists():
            print(f"[{mode}] ไม่พบ {weight} -> รัน 03_train.py ก่อน")
            continue
        source = args.source or str(ROOT / "datasets" / f"yolo_{mode}" / "test" / "images")

        model = YOLO(str(weight))
        # save=True -> ultralytics วาดผลลงภาพแล้วเซฟให้อัตโนมัติ
        results = model.predict(source=source, conf=args.conf, save=True,
                                project=str(ROOT / "runs"), name=f"predict_{mode}", exist_ok=True)

        # พิมพ์สรุปผลทีละภาพ: ชื่อไฟล์ -> คลาสที่เจอ + ความมั่นใจ
        print(f"\n===== {mode.upper()} =====")
        for r in results:
            found = [f"{model.names[int(c)]}({float(s):.2f})" for c, s in zip(r.boxes.cls, r.boxes.conf)]
            print(f"  {Path(r.path).name:<20} -> {', '.join(found) or '-'}")
        print(f"saved images -> runs/predict_{mode}/")


if __name__ == "__main__":
    main()
