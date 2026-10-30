// =====================================================================
//  infer.js : รัน YOLO "ในเบราว์เซอร์" ด้วย ONNX Runtime Web (ไม่ต้องมี server)
// =====================================================================
//  ใช้ตอน deploy เป็นเว็บ static (Hugging Face Static Space / GitHub Pages)
//  ซึ่งไม่มี Python ให้รัน -> เลยแปลงโมเดลเป็นไฟล์ .onnx (scripts/06_export_onnx.py)
//  แล้วให้เบราว์เซอร์ของคนที่เปิดเว็บเป็นคนคำนวณเอง
//
//  ขั้นตอนเหมือนที่ ultralytics ทำใน Python ทุกอย่าง:
//    1) letterbox : ย่อภาพให้ด้านยาว = 512 แล้วเติมขอบสีเทาให้เป็นสี่เหลี่ยมจัตุรัส 512x512
//    2) แปลงเป็น tensor : RGB, ค่า 0-1, เรียงแบบ [1, 3, 512, 512] (CHW)
//    3) รันโมเดล -> output0 = [1, 4 + จำนวนคลาส (+32), 5376]
//         4 ค่าแรก = กล่อง (cx, cy, w, h), ต่อด้วยคะแนนของแต่ละคลาส
//         (โมเดล seg มีอีก 32 ค่า = สัมประสิทธิ์ของ mask)
//    4) กรองคะแนน < conf ออก แล้วทำ NMS (ตัดกล่องที่ซ้อนกันเกิน)
//    5) แปลงพิกัดกลับเป็นพิกัดของภาพจริง (หักขอบเทา + หารอัตราย่อ)
//    6) (seg) mask = sigmoid(สัมประสิทธิ์ · protos) แล้วตัดเฉพาะในกล่อง
//
//  ผลลัพธ์คืนเป็น JSON รูปแบบเดียวกับ /api/predict ของหลังบ้าน
//  -> app.js ใช้โค้ดวาดผลชุดเดียวกันได้ทั้งสองโหมด
// =====================================================================

const Infer = (() => {
  const ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  const IMGSZ = 512;                       // ต้องตรงกับ imgsz ตอน export
  const CLASS_NAMES = ["airplane", "boat", "car", "motorcycle"];
  const NC = CLASS_NAMES.length;
  const IOU_THRES = 0.7;                   // ค่าเดียวกับ default ของ ultralytics
  const sessions = {};                     // โหลดโมเดลครั้งเดียวแล้วเก็บไว้

  // โหลดไฟล์ models/poly.onnx (ครั้งแรกจะช้าหน่อย ~11 MB)
  function getSession(mode) {
    if (!sessions[mode]) {
      ort.env.wasm.wasmPaths = ORT_CDN;    // ไฟล์ .wasm ของ ONNX Runtime โหลดจาก CDN
      sessions[mode] = ort.InferenceSession.create(`models/${mode}.onnx`, { executionProviders: ["wasm"] });
    }
    return sessions[mode];
  }

  // ---------- 1) + 2) letterbox แล้วแปลงเป็น tensor ----------
  function preprocess(img) {
    const w0 = img.naturalWidth, h0 = img.naturalHeight;
    const scale = Math.min(IMGSZ / w0, IMGSZ / h0);
    const nw = Math.round(w0 * scale), nh = Math.round(h0 * scale);
    const padX = Math.round((IMGSZ - nw) / 2 - 0.1);   // สูตรเดียวกับ ultralytics LetterBox
    const padY = Math.round((IMGSZ - nh) / 2 - 0.1);

    const c = document.createElement("canvas");
    c.width = c.height = IMGSZ;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "rgb(114,114,114)";                 // สีเทาที่ YOLO ใช้เติมขอบ
    ctx.fillRect(0, 0, IMGSZ, IMGSZ);
    ctx.drawImage(img, padX, padY, nw, nh);
    const rgba = ctx.getImageData(0, 0, IMGSZ, IMGSZ).data;

    // RGBA (เรียง pixel ต่อ pixel)  ->  RGB แยกเป็น 3 แผ่น (CHW) และหาร 255
    const area = IMGSZ * IMGSZ;
    const data = new Float32Array(3 * area);
    for (let p = 0; p < area; p++) {
      data[p] = rgba[p * 4] / 255;                // R
      data[area + p] = rgba[p * 4 + 1] / 255;     // G
      data[2 * area + p] = rgba[p * 4 + 2] / 255; // B
    }
    return { tensor: new ort.Tensor("float32", data, [1, 3, IMGSZ, IMGSZ]), scale, padX, padY, nw, nh, w0, h0 };
  }

  // IoU = พื้นที่ซ้อนกัน / พื้นที่รวม  (ใช้ใน NMS)
  function iou(a, b) {
    const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
    const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
    const inter = ix * iy;
    return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter + 1e-9);
  }

  // ---------- หลัก: ทำนาย 1 ภาพ ----------
  async function predict(img, mode, conf) {
    const session = await getSession(mode);
    const pre = preprocess(img);
    const t0 = performance.now();
    const outputs = await session.run({ [session.inputNames[0]]: pre.tensor });
    const ms = performance.now() - t0;

    const out0 = outputs[session.outputNames[0]];     // [1, C, N]
    const [, C, N] = out0.dims;
    const o = out0.data;
    const hasMask = C === 4 + NC + 32;

    // ---------- 4) กรองตาม conf ----------
    let cands = [];
    for (let i = 0; i < N; i++) {
      let best = 0, cls = 0;
      for (let k = 0; k < NC; k++) {
        const s = o[(4 + k) * N + i];
        if (s > best) { best = s; cls = k; }
      }
      if (best < conf) continue;
      const cx = o[i], cy = o[N + i], w = o[2 * N + i], h = o[3 * N + i];
      cands.push({ i, cls, score: best, x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2 });
    }

    // ---------- NMS: เรียงคะแนนมาก -> น้อย ตัดกล่องคลาสเดียวกันที่ซ้อนกันเกิน IOU_THRES ----------
    cands.sort((a, b) => b.score - a.score);
    const keep = [];
    for (const c of cands) {
      if (keep.every((k) => k.cls !== c.cls || iou(k, c) < IOU_THRES)) keep.push(c);
      if (keep.length >= 300) break;
    }

    // ---------- 6) mask (เฉพาะโมเดล seg) ----------
    let protos, mh, mw;
    if (hasMask) {
      const out1 = outputs[session.outputNames[1]];   // [1, 32, 128, 128]
      [, , mh, mw] = out1.dims;
      protos = out1.data;
    }

    // ---------- 5) แปลงพิกัดกลับเป็นของภาพจริง ----------
    const clampX = (v) => Math.min(Math.max(v, 0), pre.w0);
    const clampY = (v) => Math.min(Math.max(v, 0), pre.h0);
    const detections = keep.map((k) => {
      const det = {
        class_id: k.cls,
        class_name: CLASS_NAMES[k.cls],
        confidence: +k.score.toFixed(4),
        box: [clampX((k.x1 - pre.padX) / pre.scale), clampY((k.y1 - pre.padY) / pre.scale),
              clampX((k.x2 - pre.padX) / pre.scale), clampY((k.y2 - pre.padY) / pre.scale)].map((v) => +v.toFixed(1)),
      };
      if (hasMask) det.mask = buildMask(o, N, k, protos, mh, mw, pre);
      return det;
    });

    return {
      mode, engine: "browser",
      image_width: pre.w0, image_height: pre.h0,
      inference_ms: +ms.toFixed(1),
      count: detections.length,
      top_class: detections[0] ? detections[0].class_name : null,
      detections,
    };
  }

  // สร้าง mask ของวัตถุ 1 ชิ้น เป็น canvas ขาว-โปร่งใส ขนาด 128x128
  // + บอกว่าส่วนไหนของ canvas ตรงกับภาพจริง (src) เพื่อให้ app.js ยืดวาดทับภาพได้
  function buildMask(o, N, k, protos, mh, mw, pre) {
    const coef = new Float32Array(32);
    for (let j = 0; j < 32; j++) coef[j] = o[(4 + NC + j) * N + k.i];
    const r = mw / IMGSZ;                                  // 512 -> 128 = ย่อ 4 เท่า
    const bx1 = Math.floor(k.x1 * r), by1 = Math.floor(k.y1 * r);
    const bx2 = Math.ceil(k.x2 * r), by2 = Math.ceil(k.y2 * r);

    const c = document.createElement("canvas");
    c.width = mw; c.height = mh;
    const ctx = c.getContext("2d");
    const im = ctx.createImageData(mw, mh);
    const area = mw * mh;
    for (let y = Math.max(0, by1); y < Math.min(mh, by2); y++) {     // คิดเฉพาะในกล่อง
      for (let x = Math.max(0, bx1); x < Math.min(mw, bx2); x++) {
        const p = y * mw + x;
        let v = 0;
        for (let j = 0; j < 32; j++) v += coef[j] * protos[j * area + p];
        if (v > 0) {                    // sigmoid(v) > 0.5  เท่ากับ  v > 0
          im.data[p * 4] = im.data[p * 4 + 1] = im.data[p * 4 + 2] = 255;
          im.data[p * 4 + 3] = 255;
        }
      }
    }
    ctx.putImageData(im, 0, 0);
    // ส่วนของ mask ที่ตรงกับภาพจริง (ไม่รวมขอบเทาจาก letterbox)
    return { canvas: c, src: [pre.padX * r, pre.padY * r, pre.nw * r, pre.nh * r] };
  }

  return { predict };
})();
