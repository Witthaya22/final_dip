# -*- coding: utf-8 -*-
"""
=====================================================================
 STEP 2 : แปลงไฟล์ JSON จาก AnyLabeling  ->  รูปแบบ YOLO (.txt)
=====================================================================

ทำไมต้องแปลง?
    AnyLabeling เซฟ label เป็น JSON (พิกัดหน่วย pixel)
    แต่ YOLO ต้องการไฟล์ .txt หนึ่งไฟล์ต่อหนึ่งภาพ และพิกัดต้อง "normalize" (0 - 1)

รูปแบบ YOLO มี 2 แบบ ตามโจทย์:
    (1) Rectangle  (Object Detection)     ->  1 บรรทัด = 1 วัตถุ
            <class_id> <x_center> <y_center> <width> <height>
    (2) Polygon    (Instance Segmentation) ->  1 บรรทัด = 1 วัตถุ
            <class_id> <x1> <y1> <x2> <y2> <x3> <y3> ...

    ทุกค่าหารด้วยความกว้าง/สูงของภาพแล้ว เช่น x=320 ในภาพกว้าง 640 -> 0.5

ไฟล์นี้ทำอะไรอีก?
    แบ่งข้อมูลเป็น train / val / test  (ค่าเริ่มต้น 70% / 20% / 10%) แยกตามคลาส
    เพื่อให้ทุกชุดมีครบทุกคลาสในสัดส่วนเท่ากัน

วิธีใช้ (รันจากโฟลเดอร์ final_dip/):
    python scripts/02_anylabeling_to_yolo.py --mode rect --src datasets/anylabeling_rectangle --dst datasets/yolo_rect
    python scripts/02_anylabeling_to_yolo.py --mode poly --src datasets/anylabeling_polygon   --dst datasets/yolo_poly

    *** ถ้าไปวาด label เองใน AnyLabeling ก็ใช้สคริปต์นี้แปลงได้เหมือนกัน ***
    แค่ชี้ --src ไปที่โฟลเดอร์ที่มีภาพ + json ของเรา
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from ultralytics.data.converter import merge_multi_segment  # ใช้รวม polygon หลายชิ้นเป็นชิ้นเดียว

# ลำดับคลาส -> class id  (ต้องตรงกับ names: ใน rect.yaml / poly.yaml)
CLASSES = ["airplane", "boat", "car", "motorcycle"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Convert AnyLabeling JSON to YOLO txt")
    p.add_argument("--mode", choices=["rect", "poly"], required=True,
                   help="rect = กรอบสี่เหลี่ยม (detect) / poly = polygon (segment)")
    p.add_argument("--src", required=True, help="โฟลเดอร์ที่มีภาพ + json (ค้นหาทุกโฟลเดอร์ย่อย)")
    p.add_argument("--dst", required=True, help="โฟลเดอร์ปลายทางแบบ YOLO")
    p.add_argument("--split", default="0.7,0.2,0.1", help="สัดส่วน train,val,test")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def clamp01(v):
    """บังคับค่าให้อยู่ในช่วง 0..1 (กันจุดที่วาดเลยขอบภาพ)"""
    return min(max(v, 0.0), 1.0)


def shape_points(shape):
    """คืนค่าจุดของ shape เป็น [[x, y], ...]  (rectangle 2 จุด -> ขยายเป็น 4 มุม)"""
    pts = shape["points"]
    if shape.get("shape_type") == "rectangle":
        (x1, y1), (x2, y2) = pts[0], pts[1]
        pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return pts


def group_objects(shapes):
    """
    รวม shape ที่เป็น "วัตถุเดียวกัน" เข้าด้วยกัน
      - shape ที่ group_id = None  -> 1 shape = 1 วัตถุ
      - shape ที่ group_id เท่ากัน -> เป็นชิ้นส่วนของวัตถุเดียวกัน (เช่น รถที่โดนเสาบังจนขาดเป็น 2 ท่อน)
    คืนค่า: [ (label, [ชิ้นส่วน1_points, ชิ้นส่วน2_points, ...]), ... ]
    """
    objects, groups = [], {}
    for sh in shapes:
        label = sh["label"].strip().lower()
        gid = sh.get("group_id")
        if gid is None:
            objects.append((label, [shape_points(sh)]))
        else:
            key = (label, gid)
            if key not in groups:
                groups[key] = (label, [])
                objects.append(groups[key])
            groups[key][1].append(shape_points(sh))
    return objects


def object_to_yolo_line(label, parts, w, h, mode):
    """
    แปลงวัตถุ 1 ชิ้น -> 1 บรรทัด YOLO
    คืนค่า None ถ้าแปลงไม่ได้ (เช่น ชื่อ label ไม่อยู่ใน CLASSES)
    """
    if label not in CLASSES:
        print(f"    [skip] unknown label '{label}'")
        return None
    cls_id = CLASSES.index(label)
    all_pts = [p for part in parts for p in part]
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]

    if mode == "rect":
        # ----- Rectangle: หา min/max ของทุกจุด -> กรอบสี่เหลี่ยม -----
        # (ถ้า label เดิมเป็น polygon ก็แปลงเป็นกรอบได้ด้วยวิธีนี้)
        x1, x2 = clamp01(min(xs) / w), clamp01(max(xs) / w)
        y1, y2 = clamp01(min(ys) / h), clamp01(max(ys) / h)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return None
        xc, yc = x1 + bw / 2, y1 + bh / 2          # YOLO ใช้ "จุดกึ่งกลาง" ของกรอบ
        return f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

    # ----- Polygon -----
    # YOLO segment: 1 บรรทัดมีได้แค่ 1 polygon  ->  ถ้ามีหลายชิ้นส่วน ต้องเชื่อมเป็นเส้นเดียว
    # ใช้ฟังก์ชันของ ultralytics ที่เชื่อมจุดที่ใกล้กันที่สุดของแต่ละชิ้นด้วยเส้นบาง ๆ
    if len(parts) > 1:
        merged = merge_multi_segment([[c for p in part for c in p] for part in parts])
        pts = np.concatenate(merged, axis=0).tolist()
    else:
        pts = parts[0]
    if len(pts) < 3:
        return None
    coords = []
    for x, y in pts:
        coords += [f"{clamp01(x / w):.6f}", f"{clamp01(y / h):.6f}"]   # normalize 0..1
    return f"{cls_id} " + " ".join(coords)


def main():
    args = parse_args()
    src, dst = Path(args.src), Path(args.dst)
    ratios = [float(r) for r in args.split.split(",")]

    # 1) หาคู่ (ภาพ, json) ทั้งหมด แล้วจัดกลุ่มตามคลาสหลักของภาพ (ไว้แบ่ง split ให้สมดุล)
    groups = defaultdict(list)
    for js_path in sorted(src.rglob("*.json")):
        data = json.loads(js_path.read_text(encoding="utf-8"))
        img_path = js_path.parent / data["imagePath"]
        if not img_path.exists():  # เผื่อ imagePath เป็น path เต็ม -> ใช้ชื่อเดียวกับ json แทน
            img_path = next((js_path.with_suffix(e) for e in IMG_EXTS if js_path.with_suffix(e).exists()), None)
        if img_path is None or not data["shapes"]:
            print(f"    [skip] {js_path.name}: ไม่มีภาพ หรือ ไม่มี label")
            continue
        main_label = data["shapes"][0]["label"].strip().lower()
        groups[main_label].append((img_path, data))

    # 2) ลบโฟลเดอร์ปลายทางเดิม แล้วสร้างโครง train/val/test ใหม่
    if dst.exists():
        shutil.rmtree(dst)
    for s in ("train", "val", "test"):
        (dst / s / "images").mkdir(parents=True)
        (dst / s / "labels").mkdir(parents=True)

    # 3) สุ่มแบ่ง split ต่อคลาส แล้วเขียนไฟล์
    rng = random.Random(args.seed)
    stats = defaultdict(lambda: defaultdict(int))  # stats[split][class] = จำนวน object
    for label, items in sorted(groups.items()):
        rng.shuffle(items)
        n = len(items)
        n_train = round(n * ratios[0])
        n_val = round(n * ratios[1])
        splits = {
            "train": items[:n_train],
            "val": items[n_train:n_train + n_val],
            "test": items[n_train + n_val:],
        }
        for split, split_items in splits.items():
            for img_path, data in split_items:
                w, h = data["imageWidth"], data["imageHeight"]
                objs = group_objects(data["shapes"])
                lines = [ln for lb, parts in objs if (ln := object_to_yolo_line(lb, parts, w, h, args.mode))]
                # คัดลอกภาพ + เขียน label (.txt ชื่อเดียวกับภาพ)
                shutil.copy2(img_path, dst / split / "images" / img_path.name)
                (dst / split / "labels" / (img_path.stem + ".txt")).write_text("\n".join(lines) + "\n")
                for ln in lines:
                    stats[split][CLASSES[int(ln.split()[0])]] += 1

    # 4) สรุปผล
    print(f"\nYOLO dataset ({args.mode}) -> {dst}")
    print(f"{'class':<12}" + "".join(f"{s:>8}" for s in ("train", "val", "test", "total")))
    for c in CLASSES:
        row = [stats[s][c] for s in ("train", "val", "test")]
        print(f"{c:<12}" + "".join(f"{v:>8}" for v in row + [sum(row)]))


if __name__ == "__main__":
    main()
