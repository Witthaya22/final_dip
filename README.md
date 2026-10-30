# final_dip : เทรน YOLO แบบ Rectangle และ Polygon + เว็บสำหรับทดสอบ

**🌐 เว็บออนไลน์:** https://witthaya22-final-dip.static.hf.space  (หน้า Space: https://huggingface.co/spaces/witthaya22/final-dip)

**💻 โค้ด (GitHub):** https://github.com/Witthaya22/final_dip

**โจทย์:** เทรน YOLO ให้ได้ 2 แบบ คือ Rectangle และ Polygon ใช้อย่างน้อย 4 label และต้องมี object ที่ใช้เทรนอย่างน้อย 100 ชิ้นต่อ label จากนั้นส่ง Weight, ผลการเทรน และผลการ Predict

| | |
|---|---|
| คลาส (4) | `airplane` เครื่องบิน · `boat` เรือ · `car` รถยนต์ · `motorcycle` มอเตอร์ไซค์ |
| ภาพ | 600 ภาพ (คลาสละ 150) จาก COCO 2017 คัดเฉพาะภาพที่มีคลาสเดียวและวัตถุชัด |
| Rectangle | `yolo26n.pt` + `rect.yaml` (task = detect) |
| Polygon | `yolo26n-seg.pt` + `poly.yaml` (task = segment) |
| เว็บ | FastAPI (หลังบ้าน) + HTML/JS (หน้าบ้าน) อัปโหลดภาพแล้วดูผลการทายด้วยโมเดล **Polygon** (เว็บใช้แค่ Polygon, Rectangle ใช้ส่งงาน) |

## จำนวน object (โจทย์กำหนดให้มี ≥ 100 ต่อ label ในชุดที่ใช้เทรน)

| class | train | val | test | total |
|---|---:|---:|---:|---:|
| airplane | 135 | 43 | 18 | 196 |
| boat | 153 | 48 | 35 | 236 |
| car | 161 | 49 | 28 | 238 |
| motorcycle | 154 | 37 | 16 | 207 |

ทั้งแบบ Rectangle และ Polygon ได้ตัวเลขเท่ากัน เพราะใช้ object ชุดเดียวกัน ต่างกันแค่รูปแบบ label

## ผลการเทรน (ชุด test)

เทรน 20 epoch, imgsz 512, บน CPU (Intel i5-1334U) · Rectangle ใช้เวลา ~40 นาที, Polygon ~50 นาที

| โมเดล | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|
| Rectangle (`yolo26n`) | **0.811** | 0.586 | – | – |
| Polygon (`yolo26n-seg`) | 0.791 | 0.557 | **0.798** | 0.468 |

mAP50 = ทายตำแหน่งถูก (ซ้อนทับ ≥ 50%) และคลาสถูก โดยเฉลี่ยทุกคลาส · ยิ่งใกล้ 1 ยิ่งดี

## โครงสร้างโฟลเดอร์

```
final_dip/
├── rect.yaml / poly.yaml          ไฟล์ตั้งค่า dataset ของแต่ละแบบ
├── lab_final_rect.ipynb           เทรน Rectangle (โครงเดียวกับ lab_2box.ipynb)
├── lab_final_poly.ipynb           เทรน Polygon + export ให้เว็บ (โครงเดียวกับ lab_2.ipynb)
├── lab_final_web.ipynb            อธิบายเว็บทีละขั้น: หลังบ้าน, หน้าบ้าน, รันโมเดลในเบราว์เซอร์, deploy
├── scripts/
│   ├── 01_build_dataset_from_coco.py   ดึงภาพและ label จาก COCO แล้วเซฟเป็น JSON แบบ AnyLabeling
│   ├── 02_anylabeling_to_yolo.py       แปลง JSON เป็น YOLO txt และแบ่ง train/val/test
│   ├── 03_train.py                     เทรน rect และ poly
│   ├── 04_predict.py                   predict ภาพ test แล้วเซฟภาพผลลัพธ์
│   ├── 05_deploy_huggingface.py        deploy เว็บขึ้น Hugging Face Spaces (static, ฟรี)
│   ├── 06_export_onnx.py               แปลง .pt เป็น .onnx ให้เบราว์เซอร์รันได้
│   └── check_status.py                 เช็คว่าเทรนเสร็จหรือยัง
├── datasets/
│   ├── raw_images/<class>/             ภาพเปล่า แยกโฟลเดอร์ตามคลาส (เอาไปวาด label เองได้)
│   ├── anylabeling_rectangle/<class>/  ภาพ + JSON แบบ rectangle (เปิดใน AnyLabeling ได้เลย)
│   ├── anylabeling_polygon/<class>/    ภาพ + JSON แบบ polygon (เปิดใน AnyLabeling ได้เลย)
│   ├── yolo_rect/ , yolo_poly/         สร้างจากสคริปต์ 02 (train/val/test)
│   └── SOURCES.json                    แหล่งที่มาของแต่ละภาพ (COCO, CC BY 4.0)
├── runs/
│   ├── rect/ , poly/                   ผลการเทรน: weights/best.pt, results.png, confusion_matrix ฯลฯ
│   ├── rect_test/ , poly_test/         ผลประเมินบนชุด test
│   └── predict_rect/ , predict_poly/   ภาพผลการ predict
├── weights/rect_best.pt , poly_best.pt   weight ที่ส่งงาน และเป็นตัวที่เว็บโหลดไปใช้
├── web/
│   ├── backend/app.py              REST API (FastAPI)
│   └── frontend/                   index.html, app.js, infer.js (รันโมเดลในเบราว์เซอร์), style.css, config.js, models/poly.onnx
├── Dockerfile                      สำหรับ deploy
└── PRESENTATION_SCRIPT.md          สคริปต์อัดคลิปนำเสนอ
```

## Notebook เทรน : `lab_final_rect.ipynb` และ `lab_final_poly.ipynb`

แยกเป็นสองไฟล์ตามแบบของ lab รันจริงไว้แล้วและเก็บผลทุก cell ไว้ในไฟล์ ก่อนทุก cell มีคำอธิบายว่าโค้ดทำอะไร อ่านตอนอัดคลิปได้เลย

| ขั้น | Cell | ทำอะไร |
|---|---|---|
| 0 | 1–3 | import, ตั้งค่า, `ultralytics.checks()` |
| 1 | 4–5 | ดูไฟล์ yaml, นับภาพและ object ต่อคลาส, วาด label จากไฟล์ txt กลับลงภาพ |
| 2 | 6–7 | โหลดโมเดล pretrained แล้วลองทายก่อนเทรน |
| 3 | 8 | **เทรน** (อธิบายทุกพารามิเตอร์และค่า loss) |
| 4 | 9–11 | กราฟผลการเทรน, confusion matrix, mAP ชุด test แยกคลาส |
| 5 | 12–14 | predict คลาสละ 1 ภาพ, นับว่าทายถูกกี่ภาพ, เซฟผล predict ทั้งชุด test |
| 6 | 15–17 (เฉพาะ poly) | **export ให้เว็บ:** `.pt` สำหรับหลังบ้าน, `.onnx` สำหรับหน้าเว็บ, ตรวจไฟล์ ONNX, deploy ขึ้น Hugging Face |

## Notebook เว็บ : `lab_final_web.ipynb`

อธิบายเว็บทีละขั้น (มีคำอธิบายก่อนทุก cell และรันไว้แล้ว)

| Part | Cell | อธิบาย |
|---|---|---|
| 1 หลังบ้าน | 2–6 | โค้ด `/api/predict`, เปิด server, `GET /api/health`, `POST /api/predict` และความหมายของ JSON, วาดผลจาก JSON แบบ `app.js` |
| 2 หน้าบ้าน | 7–8 | โครง `index.html`, ลำดับการทำงานของ `app.js`, `config.js` (โหมด api / browser) |
| 3 รันในเบราว์เซอร์ | 9–14 | ขั้นตอนของ `infer.js` เขียนเป็น Python: letterbox → tensor → รันโมเดล → conf+NMS → แปลงพิกัด → สร้าง mask, เทียบผลกับ ultralytics (ตรงกัน) |
| 4 Deploy | 15–16 | Hugging Face static Space, สคริปต์ deploy, เช็คเว็บออนไลน์ |

## สิ่งที่ต้องส่ง อยู่ตรงไหน

| สิ่งที่ส่ง | ไฟล์ |
|---|---|
| Weight | `weights/rect_best.pt`, `weights/poly_best.pt` |
| ผลการเทรน | `runs/rect/` และ `runs/poly/` (`results.png`, `confusion_matrix*.png`, `*_curve.png`, `results.csv`) |
| ผลการ Predict | `runs/predict_rect/`, `runs/predict_poly/` |

## วิธีรัน (ให้รันจากในโฟลเดอร์ `final_dip/`)

```bash
pip install -r requirements.txt

# (ข้ามได้ เพราะมีภาพเตรียมไว้แล้ว) ดึงภาพจาก COCO
python scripts/01_build_dataset_from_coco.py --ann path/to/instances_train2017.json

# แปลง label เป็นรูปแบบ YOLO
python scripts/02_anylabeling_to_yolo.py --mode rect --src datasets/anylabeling_rectangle --dst datasets/yolo_rect
python scripts/02_anylabeling_to_yolo.py --mode poly --src datasets/anylabeling_polygon   --dst datasets/yolo_poly

# เทรน (บน CPU ใช้เวลาประมาณ 1–1.5 ชั่วโมง)
python scripts/03_train.py --mode both

# predict ภาพในชุด test
python scripts/04_predict.py

# เปิดเว็บในเครื่อง แล้วเข้า http://localhost:8000  (หน้าเอกสาร API อยู่ที่ /docs)
uvicorn web.backend.app:app --port 8000
```

### ถ้าจะวาด label เองใน AnyLabeling
1. เปิดโฟลเดอร์ `datasets/raw_images/car` (หรือคลาสอื่น) ใน AnyLabeling
2. วาดด้วย Rectangle หรือ Polygon และตั้งชื่อ label ให้ตรงกับ `airplane / boat / car / motorcycle` (ตัวพิมพ์เล็ก)
3. ถ้าวัตถุชิ้นเดียวถูกบังจนแยกเป็นหลายส่วน ให้ใส่ group id เดียวกันทุกส่วน
4. ใช้สคริปต์ 02 แปลง โดยชี้ `--src datasets/raw_images`

## Deploy ขึ้น server

ตอนนี้ขึ้นแล้วที่ **https://witthaya22-final-dip.static.hf.space**

**ทำงานยังไง:** Hugging Face แบบฟรีไม่ให้รัน server Python แล้ว (Docker Space ต้องสมัคร PRO) จึงใช้ **Static Space** ที่ฟรีแทน
- แปลงโมเดลเป็นไฟล์ `.onnx` (`scripts/06_export_onnx.py`)
- หน้าเว็บโหลดโมเดลแล้วรันในเบราว์เซอร์ของคนที่เปิดเว็บ ด้วย ONNX Runtime Web (`web/frontend/infer.js`)
- ขั้นตอนเหมือน YOLO ใน Python ทุกอย่าง: letterbox → โมเดล → กรอง conf → NMS → แปลงพิกัดกลับ → (seg) สร้าง mask
- หน้าเว็บเลือกเองอัตโนมัติ (`config.js`): ถ้าเจอหลังบ้าน FastAPI จะใช้ API ถ้าไม่เจอจะรันในเบราว์เซอร์

**อัปเดตเว็บหลังเทรนใหม่**
```bash
python scripts/06_export_onnx.py
python scripts/05_deploy_huggingface.py --space witthaya22/final-dip
```

**ถ้าต้องการหลังบ้านจริงบน server** (เช่น สมัคร HF PRO, Render หรือ VPS) ใช้ `Dockerfile` ได้เลย
```bash
docker build -t final-dip .
docker run -p 7860:7860 final-dip
```

## หมายเหตุ
- weight ตั้งต้นของ YOLO ถูกฝึกมาจาก COCO train2017 ซึ่งเป็นชุดเดียวกับที่เราดึงภาพมา ผลบนชุด test จึงอาจออกมาดีกว่าความเป็นจริง วิธีทดสอบที่ตรงที่สุดคืออัปโหลดภาพที่ถ่ายเองหรือหาจากเน็ตผ่านหน้าเว็บ
- ใน `custom.yaml` และ `custom2.yaml` ของ lab เดิมเขียนคีย์ว่า `name:` แต่ YOLO อ่านคีย์ `names:` (มี s) ใน `rect.yaml` และ `poly.yaml` แก้ให้ถูกแล้ว
- `lab_2.ipynb` เดิมเทรน label polygon ด้วยโมเดล detect (`yolo26n.pt`) ทำให้ได้ผลเป็นกรอบสี่เหลี่ยมเท่านั้น ถ้าต้องการผลแบบ polygon ต้องใช้ `yolo26n-seg.pt` อย่างที่ทำในงานนี้
