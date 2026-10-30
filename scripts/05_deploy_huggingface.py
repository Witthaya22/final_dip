import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "web" / "frontend"

SPACE_README = """---
title: final_dip YOLO Vehicle Detector
emoji: 🚗
colorFrom: indigo
colorTo: red
sdk: static
app_file: index.html
pinned: false
---

# YOLO Vehicle Detector (final_dip)
อัปโหลดภาพ แล้ว YOLO จะทายว่าเป็น airplane / boat / car / motorcycle
ด้วยโมเดล Polygon (segment)
โมเดลรันในเบราว์เซอร์ด้วย ONNX Runtime Web
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--space", required=True, help="เช่น myname/final-dip")
    args = p.parse_args()

    assert (FRONTEND / "models" / "poly.onnx").exists(), "ไม่พบ models/poly.onnx -> รัน 06_export_onnx.py ก่อน"

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="static", exist_ok=True)

    api.upload_folder(repo_id=args.space, repo_type="space", folder_path=str(FRONTEND),
                      delete_patterns=["*.html", "*.js", "*.css", "samples/*", "models/*"],
                      commit_message="deploy final_dip web")
    api.upload_file(repo_id=args.space, repo_type="space", path_in_repo="README.md",
                    path_or_fileobj=SPACE_README.encode("utf-8"))

    user, name = args.space.split("/")
    print("\nอัปโหลดเสร็จ! รอประมาณ 1 นาที แล้วเปิด:")
    print(f"  หน้า Space : https://huggingface.co/spaces/{args.space}")
    print(f"  เว็บตรง ๆ  : https://{user.lower()}-{name.lower().replace('_', '-')}.static.hf.space")


if __name__ == "__main__":
    main()
