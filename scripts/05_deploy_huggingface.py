# -*- coding: utf-8 -*-
"""
=====================================================================
 STEP 7 : Deploy เว็บขึ้น Hugging Face Spaces (ฟรี) ด้วยคำสั่งเดียว
=====================================================================

ทำไมเป็น "Static Space"?
    Hugging Face ให้ใช้ Space แบบ Docker/Python ฟรีไม่ได้แล้ว (ต้องสมัคร PRO)
    แต่ Space แบบ Static (มีแค่ไฟล์เว็บ) ยังใช้ฟรี
    เราเลยอัปโหลดหน้าเว็บ + โมเดล .onnx ขึ้นไป แล้วให้เบราว์เซอร์ของคนที่เปิดเว็บ
    เป็นคนรันโมเดลเอง (web/frontend/infer.js)
    ข้อดี: ฟรี, ไม่ต้องมี server, เว็บไม่หลับ, ส่งลิงก์ให้ใครก็ใช้ได้ทันที

ขั้นตอน (ทำครั้งเดียว):
    1) สมัครบัญชี https://huggingface.co
    2) สร้าง Access Token (สิทธิ์ Write) ที่ https://huggingface.co/settings/tokens
    3) pip install huggingface_hub
    4) hf auth login            (วาง token ที่ได้จากข้อ 2)

แล้วรัน (จากโฟลเดอร์ final_dip/):
    python scripts/06_export_onnx.py                                  # แปลงโมเดลเป็น .onnx ก่อน
    python scripts/05_deploy_huggingface.py --space <username>/final-dip

ได้เว็บที่:  https://<username>-final-dip.static.hf.space
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "web" / "frontend"

# README ของ Space ต้องมี YAML header นี้ เพื่อบอก HF ว่าเป็นเว็บ static และหน้าแรกคือ index.html
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
    # 1) สร้าง Space แบบ static (exist_ok=True -> ถ้ามีอยู่แล้วก็ใช้ของเดิม)
    api.create_repo(args.space, repo_type="space", space_sdk="static", exist_ok=True)

    # 2) อัปโหลดทุกไฟล์ใน web/frontend ไปไว้ที่ root ของ Space
    #    (index.html, app.js, infer.js, style.css, config.js, models/*.onnx)
    #    delete_patterns = ลบไฟล์เว็บเก่าบน Space ที่ไม่มีในเครื่องแล้ว (ให้บน Space ตรงกับในเครื่องเสมอ)
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
