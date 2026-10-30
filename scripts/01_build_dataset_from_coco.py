import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests

CLASSES = ["airplane", "boat", "car", "motorcycle"]

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description="Build 4-class dataset from COCO 2017")
    p.add_argument("--ann", required=True, help="path ของ instances_train2017.json")
    p.add_argument("--out", default=str(ROOT / "datasets"), help="โฟลเดอร์ปลายทาง")
    p.add_argument("--per-class", type=int, default=150, help="จำนวนภาพต่อคลาส")
    p.add_argument("--max-objects", type=int, default=6, help="จำนวนวัตถุสูงสุดต่อภาพ")
    p.add_argument("--seed", type=int, default=42, help="seed สำหรับสุ่ม (ได้ผลเหมือนเดิมทุกครั้ง)")
    return p.parse_args()


def select_images(coco, per_class, max_objects, seed):
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"] if c["name"] in CLASSES}

    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        if a["category_id"] in cat_id_to_name:
            anns_by_img[a["image_id"]].append(a)

    images = {im["id"]: im for im in coco["images"]}
    candidates = defaultdict(list)

    for img_id, anns in anns_by_img.items():
        im = images[img_id]
        img_area = im["width"] * im["height"]

        names = {cat_id_to_name[a["category_id"]] for a in anns}
        if len(names) != 1:
            continue
        name = names.pop()

        if any(a["iscrowd"] for a in anns):
            continue

        if len(anns) > max_objects:
            continue

        ratios = [a["area"] / img_area for a in anns]
        if min(ratios) < 0.005:
            continue

        if max(ratios) < 0.05:
            continue

        candidates[name].append((im, anns))

    rng = random.Random(seed)
    selected = {}
    for name in CLASSES:
        pool = sorted(candidates[name], key=lambda x: x[0]["id"])
        rng.shuffle(pool)
        selected[name] = pool[:per_class]
        print(f"  {name:<11} candidates={len(pool):>5}  selected={len(selected[name])}")
    return selected


def download(url, dest, retries=3):
    if dest.exists() and dest.stat().st_size > 0:
        return True
    for i in range(retries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return True
        except Exception as e:
            print(f"    retry {i + 1}: {url} ({e})")
            time.sleep(2)
    return False


def make_anylabeling_json(file_name, im, shapes):
    return {
        "version": "0.4.43",
        "flags": {},
        "shapes": [
            {
                "label": s["label"],
                "text": "",
                "points": s["points"],
                "group_id": s.get("group_id"),
                "shape_type": s["shape_type"],
                "flags": {},
            }
            for s in shapes
        ],
        "imagePath": file_name,
        "imageData": None,
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
            jobs.append((name, f"{name}_{idx:03d}.jpg", im, anns))

    with ThreadPoolExecutor(16) as ex:
        ok = list(ex.map(lambda j: download(j[2]["coco_url"], out / "raw_images" / j[0] / j[1]), jobs))
    print(f"  downloaded {sum(ok)}/{len(jobs)} images")

    obj_count = defaultdict(int)
    for (name, fname, im, anns), good in zip(jobs, ok):
        if not good:
            continue
        img_bytes = (out / "raw_images" / name / fname).read_bytes()

        rect_shapes = []
        for a in anns:
            x, y, w, h = a["bbox"]
            rect_shapes.append({
                "label": name,
                "points": [[round(x, 1), round(y, 1)], [round(x + w, 1), round(y + h, 1)]],
                "shape_type": "rectangle",
            })

        poly_shapes = []
        for obj_idx, a in enumerate(anns):
            segs = [sg for sg in a["segmentation"] if len(sg) >= 6]
            for seg in segs:
                pts = [[round(seg[i], 1), round(seg[i + 1], 1)] for i in range(0, len(seg), 2)]
                poly_shapes.append({"label": name, "points": pts, "shape_type": "polygon",
                                    "group_id": obj_idx if len(segs) > 1 else None})

        for d, shapes in (("anylabeling_rectangle", rect_shapes), ("anylabeling_polygon", poly_shapes)):
            (out / d / name / fname).write_bytes(img_bytes)
            js = make_anylabeling_json(fname, im, shapes)
            (out / d / name / fname.replace(".jpg", ".json")).write_text(
                json.dumps(js, indent=2, ensure_ascii=False), encoding="utf-8")

        obj_count[name] += len(anns)

    print("\nObjects per class (โจทย์ต้องการ >= 100):")
    for name in CLASSES:
        print(f"  {name:<11} {obj_count[name]}")

    credits = [{"file": f"{n}/{f}", "coco_id": im["id"], "url": im["coco_url"], "license": im["license"]}
               for (n, f, im, _), g in zip(jobs, ok) if g]
    (out / "SOURCES.json").write_text(json.dumps(credits, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
