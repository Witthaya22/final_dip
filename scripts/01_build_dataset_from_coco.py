# -*- coding: utf-8 -*-
"""
=====================================================================
 STEP 1 : สร้าง Dataset จาก COCO 2017 (มี Label ให้แล้วทั้ง Box และ Polygon)
=====================================================================

ไฟล์นี้ทำอะไร?
    1) อ่านไฟล์ annotation ของ COCO (instances_train2017.json)
       ซึ่งเป็น dataset สาธารณะที่ "มีคนวาด label ไว้แล้ว" ทั้งแบบกรอบ (bbox)
       และแบบเส้นรอบวัตถุ (polygon / segmentation)
    2) เลือกเฉพาะ 4 คลาสที่เราต้องการ:  airplane, boat, car, motorcycle
       (เครื่องบิน, เรือ, รถยนต์, มอเตอร์ไซค์ -> หน้าตาต่างกันชัด แยกง่าย)
    3) คัดภาพที่ "ง่าย" คือ
         - ในภาพมีแค่คลาสเดียวจาก 4 คลาสนี้ (ไม่ปนกัน)
         - วัตถุไม่เล็กจนเกินไป และไม่เยอะจนเกินไป
    4) ดาวน์โหลดภาพ แล้วบันทึกออกมา 3 ชุด
         datasets/raw_images/<class>/            -> ภาพเปล่า ๆ (ไว้ไปวาด label เองใน AnyLabeling)
         datasets/anylabeling_rectangle/<class>/ -> ภาพ + JSON แบบ rectangle (เปิดใน AnyLabeling ได้เลย)
         datasets/anylabeling_polygon/<class>/   -> ภาพ + JSON แบบ polygon   (เปิดใน AnyLabeling ได้เลย)

    ไฟล์ JSON ที่ได้ มีรูปแบบ "เหมือนกับที่ AnyLabeling เซฟออกมา" ทุกประการ
    (version 0.4.43, shapes[].shape_type = "rectangle" / "polygon")
    ดังนั้นขั้นต่อไปใช้สคริปต์ 02 แปลงเป็น YOLO ได้เหมือนกับ label ที่เราวาดเอง

วิธีใช้:
    1) ดาวน์โหลด http://images.cocodataset.org/annotations/annotations_trainval2017.zip
       แล้วแตกไฟล์ จะได้ annotations/instances_train2017.json
    2) python scripts/01_build_dataset_from_coco.py --ann <path>/instances_train2017.json

หมายเหตุ: ไม่ต้องรันใหม่ก็ได้ เพราะใน datasets/ มีภาพที่ดาวน์โหลดไว้ให้แล้ว
"""

import argparse          # รับค่าพารามิเตอร์จาก command line
import json              # อ่าน/เขียนไฟล์ JSON
import random            # สุ่มเลือกภาพ
import time              # หน่วงเวลาเวลาดาวน์โหลดพลาด
from collections import defaultdict
from pathlib import Path  # จัดการ path ไฟล์ให้ใช้ได้ทั้ง Windows/Linux
from concurrent.futures import ThreadPoolExecutor  # ดาวน์โหลดหลายภาพพร้อมกัน

import requests          # ใช้ดาวน์โหลดภาพจากอินเทอร์เน็ต

# ---------------------------------------------------------------
# ตั้งค่าคลาสที่ต้องการ (ชื่อต้องตรงกับชื่อคลาสใน COCO)
# ลำดับในลิสต์นี้ = class id ที่จะใช้ใน YOLO (0,1,2,3)
# ---------------------------------------------------------------
CLASSES = ["airplane", "boat", "car", "motorcycle"]

# โฟลเดอร์หลักของโปรเจกต์ (final_dip/) = โฟลเดอร์แม่ของ scripts/
ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    """อ่านค่าที่ผู้ใช้ส่งเข้ามาตอนรันสคริปต์"""
    p = argparse.ArgumentParser(description="Build 4-class dataset from COCO 2017")
    p.add_argument("--ann", required=True, help="path ของ instances_train2017.json")
    p.add_argument("--out", default=str(ROOT / "datasets"), help="โฟลเดอร์ปลายทาง")
    p.add_argument("--per-class", type=int, default=150, help="จำนวนภาพต่อคลาส")
    p.add_argument("--max-objects", type=int, default=6, help="จำนวนวัตถุสูงสุดต่อภาพ")
    p.add_argument("--seed", type=int, default=42, help="seed สำหรับสุ่ม (ได้ผลเหมือนเดิมทุกครั้ง)")
    return p.parse_args()


def select_images(coco, per_class, max_objects, seed):
    """
    เลือกภาพที่ "ง่าย" ต่อคลาส
    คืนค่า: dict { class_name: [ (image_info, [annotation, ...]), ... ] }
    """
    # map: COCO category id -> ชื่อคลาส  (เฉพาะ 4 คลาสที่เราสนใจ)
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"] if c["name"] in CLASSES}

    # รวม annotation ตาม image_id  -> {image_id: [ann, ann, ...]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        if a["category_id"] in cat_id_to_name:
            anns_by_img[a["image_id"]].append(a)

    images = {im["id"]: im for im in coco["images"]}
    candidates = defaultdict(list)

    for img_id, anns in anns_by_img.items():
        im = images[img_id]
        img_area = im["width"] * im["height"]

        # (1) ในภาพต้องมีคลาสเดียวเท่านั้น -> แยกง่าย ไม่สับสน
        names = {cat_id_to_name[a["category_id"]] for a in anns}
        if len(names) != 1:
            continue
        name = names.pop()

        # (2) ห้ามมี iscrowd (กลุ่มวัตถุที่ถูกวาดเป็นก้อนเดียว ไม่ใช่ polygon ปกติ)
        if any(a["iscrowd"] for a in anns):
            continue

        # (3) จำนวนวัตถุไม่เยอะเกิน (วาดเองใน AnyLabeling ได้ไม่เหนื่อย)
        if len(anns) > max_objects:
            continue

        # (4) ทุกวัตถุต้องไม่เล็กเกินไป (>= 0.5% ของภาพ)
        #     ถ้ามีวัตถุเล็กจิ๋วแล้วเราตัดทิ้ง จะกลายเป็น "วัตถุที่ไม่ได้ label" ซึ่งทำให้โมเดลสับสน
        #     จึงตัดทั้งภาพทิ้งแทน
        ratios = [a["area"] / img_area for a in anns]
        if min(ratios) < 0.005:
            continue

        # (5) ต้องมีวัตถุหลักอย่างน้อย 1 ชิ้นที่ใหญ่พอ (>= 5% ของภาพ) -> เห็นชัด
        if max(ratios) < 0.05:
            continue

        candidates[name].append((im, anns))

    # สุ่มเลือก per_class ภาพต่อคลาส (ใช้ seed เพื่อให้ได้ชุดเดิมทุกครั้ง)
    rng = random.Random(seed)
    selected = {}
    for name in CLASSES:
        pool = sorted(candidates[name], key=lambda x: x[0]["id"])
        rng.shuffle(pool)
        selected[name] = pool[:per_class]
        print(f"  {name:<11} candidates={len(pool):>5}  selected={len(selected[name])}")
    return selected


def download(url, dest, retries=3):
    """ดาวน์โหลดไฟล์จาก url ไปเก็บที่ dest (ลองใหม่ได้ถ้าพลาด)"""
    if dest.exists() and dest.stat().st_size > 0:
        return True  # มีอยู่แล้ว ไม่ต้องโหลดซ้ำ
    for i in range(retries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"    retry {i + 1}: {url} ({e})")
            time.sleep(2)
    return False


def make_anylabeling_json(file_name, im, shapes):
    """
    สร้าง dict ที่มีโครงสร้างเหมือนไฟล์ .json ที่ AnyLabeling เซฟออกมา
    shapes = [ {label, points, shape_type}, ... ]
    """
    return {
        "version": "0.4.43",
        "flags": {},
        "shapes": [
            {
                "label": s["label"],
                "text": "",
                "points": s["points"],          # list ของ [x, y] (หน่วยเป็น pixel)
                "group_id": s.get("group_id"),  # ชิ้นส่วนของวัตถุเดียวกันจะมี group_id เดียวกัน
                "shape_type": s["shape_type"],  # "rectangle" หรือ "polygon"
                "flags": {},
            }
            for s in shapes
        ],
        "imagePath": file_name,                 # ชื่อไฟล์ภาพ (อยู่โฟลเดอร์เดียวกับ json)
        "imageData": None,                      # ไม่ฝังภาพลงใน json (ไฟล์เล็กลง)
        "imageHeight": im["height"],
        "imageWidth": im["width"],
    }


def main():
    args = parse_args()
    out = Path(args.out)

    print("[1/3] Loading COCO annotations ... (ไฟล์ใหญ่ ~450MB รอสักครู่)")
    coco = json.loads(Path(args.ann).read_text(encoding="utf-8"))

    print("[2/3] Selecting easy images per class")
    selected = select_images(coco, args.per_class, args.max_objects, args.seed)

    print("[3/3] Downloading images + writing AnyLabeling JSON")
    jobs = []
    for name, items in selected.items():
        for d in ("raw_images", "anylabeling_rectangle", "anylabeling_polygon"):
            (out / d / name).mkdir(parents=True, exist_ok=True)
        for idx, (im, anns) in enumerate(items, start=1):
            # ตั้งชื่อไฟล์ให้อ่านง่าย เช่น car_001.jpg
            jobs.append((name, f"{name}_{idx:03d}.jpg", im, anns))

    # ดาวน์โหลดภาพพร้อมกัน 16 ไฟล์ ไปไว้ที่ raw_images/<class>/
    with ThreadPoolExecutor(16) as ex:
        ok = list(ex.map(lambda j: download(j[2]["coco_url"], out / "raw_images" / j[0] / j[1]), jobs))
    print(f"  downloaded {sum(ok)}/{len(jobs)} images")

    obj_count = defaultdict(int)
    for (name, fname, im, anns), good in zip(jobs, ok):
        if not good:
            continue
        img_bytes = (out / "raw_images" / name / fname).read_bytes()

        # ---------- แบบ Rectangle ----------
        # COCO bbox = [x, y, w, h]  ->  AnyLabeling rectangle = [[x1, y1], [x2, y2]]
        rect_shapes = []
        for a in anns:
            x, y, w, h = a["bbox"]
            rect_shapes.append({
                "label": name,
                "points": [[round(x, 1), round(y, 1)], [round(x + w, 1), round(y + h, 1)]],
                "shape_type": "rectangle",
            })

        # ---------- แบบ Polygon ----------
        # COCO segmentation = [[x1, y1, x2, y2, ...], ...]  (1 วัตถุอาจมีหลายชิ้นส่วน)
        # AnyLabeling polygon = [[x1, y1], [x2, y2], ...]  -> 1 ชิ้นส่วน = 1 shape
        # ถ้าวัตถุถูกบังจนแยกเป็นหลายชิ้น (เช่น เครื่องบินโดนเสาบัง) จะใส่ group_id เดียวกัน
        # เหมือนตอนเรากด group ใน AnyLabeling -> สคริปต์ 02 จะรวมกลับเป็น 1 วัตถุ
        poly_shapes = []
        for obj_idx, a in enumerate(anns):
            segs = [sg for sg in a["segmentation"] if len(sg) >= 6]  # polygon ต้องมีอย่างน้อย 3 จุด
            for seg in segs:
                pts = [[round(seg[i], 1), round(seg[i + 1], 1)] for i in range(0, len(seg), 2)]
                poly_shapes.append({"label": name, "points": pts, "shape_type": "polygon",
                                    "group_id": obj_idx if len(segs) > 1 else None})

        # เขียนภาพ + json ลงสองโฟลเดอร์
        for d, shapes in (("anylabeling_rectangle", rect_shapes), ("anylabeling_polygon", poly_shapes)):
            (out / d / name / fname).write_bytes(img_bytes)
            js = make_anylabeling_json(fname, im, shapes)
            (out / d / name / fname.replace(".jpg", ".json")).write_text(
                json.dumps(js, indent=2, ensure_ascii=False), encoding="utf-8")

        obj_count[name] += len(anns)

    # สรุปจำนวน object ต่อคลาส (โจทย์บังคับ >= 100 object ต่อ label)
    print("\nObjects per class (โจทย์ต้องการ >= 100):")
    for name in CLASSES:
        print(f"  {name:<11} {obj_count[name]}")

    # เก็บแหล่งที่มาไว้อ้างอิง (license COCO = CC BY 4.0)
    credits = [{"file": f"{n}/{f}", "coco_id": im["id"], "url": im["coco_url"], "license": im["license"]}
               for (n, f, im, _), g in zip(jobs, ok) if g]
    (out / "SOURCES.json").write_text(json.dumps(credits, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
