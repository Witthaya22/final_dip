import os
import shutil
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "frontend" / "models"
IMGSZ = 512


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for mode in ("poly",):
        weight = ROOT / "weights" / f"{mode}_best.pt"
        if not weight.exists():
            print(f"[{mode}] ไม่พบ {weight} -> รัน 03_train.py ก่อน")
            continue
        onnx_path = YOLO(str(weight)).export(format="onnx", imgsz=IMGSZ, simplify=True)
        shutil.move(onnx_path, OUT / f"{mode}.onnx")
        size = (OUT / f"{mode}.onnx").stat().st_size / 1e6
        print(f"[{mode}] -> web/frontend/models/{mode}.onnx  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
