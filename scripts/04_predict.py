import argparse
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

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
        results = model.predict(source=source, conf=args.conf, save=True,
                                project=str(ROOT / "runs"), name=f"predict_{mode}", exist_ok=True)

        print(f"\n===== {mode.upper()} =====")
        for r in results:
            found = [f"{model.names[int(c)]}({float(s):.2f})" for c, s in zip(r.boxes.cls, r.boxes.conf)]
            print(f"  {Path(r.path).name:<20} -> {', '.join(found) or '-'}")
        print(f"saved images -> runs/predict_{mode}/")


if __name__ == "__main__":
    main()
