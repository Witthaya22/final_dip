import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOTAL_EPOCHS = 20

ready = 0
for mode in ("rect", "poly"):
    csv_path = ROOT / "runs" / mode / "results.csv"
    weight = ROOT / "weights" / f"{mode}_best.pt"

    if weight.exists():
        status = "เสร็จแล้ว พร้อมใช้"
        ready += 1
    elif csv_path.exists():
        status = "กำลังเทรน"
    else:
        status = "ยังไม่เริ่ม (รอคิว)"

    line = f"[{mode}] {status}"
    if csv_path.exists():
        rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
        if rows:
            last = {k.strip(): v for k, v in rows[-1].items()}
            line += f"  | epoch {int(float(last['epoch']))}/{TOTAL_EPOCHS}  mAP50 = {float(last['metrics/mAP50(B)']):.3f}"
    print(line)

print("\nพร้อมใช้ทั้ง 2 แบบแล้ว เปิดเว็บ / deploy ได้เลย" if ready == 2 else "\nยังไม่เสร็จทั้งหมด รอก่อน")
