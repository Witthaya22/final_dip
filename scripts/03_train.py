import argparse
import os
import shutil
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

CONFIGS = {
    "rect": ("yolo26n.pt", "rect.yaml"),
    "poly": ("yolo26n-seg.pt", "poly.yaml"),
}


def train_one(mode, epochs, imgsz, batch, device):
    pretrained, data_yaml = CONFIGS[mode]
    print(f"\n========== TRAIN {mode.upper()}  ({pretrained} + {data_yaml}) ==========")

    model = YOLO(str(ROOT / pretrained) if (ROOT / pretrained).exists() else pretrained)

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=15,
        device=device,
        workers=2,
        project=str(ROOT / "runs"),
        name=mode,
        exist_ok=True,
        seed=42,
        plots=True,
    )

    best = ROOT / "runs" / mode / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(data=data_yaml, split="test", imgsz=imgsz, device=device,
                                  project=str(ROOT / "runs"), name=f"{mode}_test", exist_ok=True)
    print(f"[{mode}] TEST box mAP50 = {metrics.box.map50:.3f}   mAP50-95 = {metrics.box.map:.3f}")
    if mode == "poly":
        print(f"[{mode}] TEST mask mAP50 = {metrics.seg.map50:.3f}   mAP50-95 = {metrics.seg.map:.3f}")

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

    os.chdir(ROOT)
    modes = ["rect", "poly"] if args.mode == "both" else [args.mode]
    for m in modes:
        train_one(m, args.epochs, args.imgsz, args.batch, args.device)


if __name__ == "__main__":
    main()
