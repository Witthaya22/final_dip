// =====================================================================
//  app.js : โค้ดหน้าบ้านทั้งหมด
// =====================================================================
//  ลำดับการทำงาน:
//    1) เปิดหน้าเว็บ -> checkHealth() ถาม API ว่าพร้อมไหม
//    2) ผู้ใช้เลือกภาพ (ลากวาง / เลือกไฟล์ / ภาพตัวอย่าง) -> setImage()
//    3) กด Predict -> runPredict() ส่งภาพไป POST /api/predict
//    4) ได้ JSON กลับมา -> renderResult() วาดกรอบ/polygon ลงบน <canvas> + ทำตารางผล
//
//  มี 2 วิธีทำนาย (เลือกอัตโนมัติใน checkHealth):
//    - "api"     : ส่งภาพไปหลังบ้าน FastAPI (ตอนรันในเครื่องด้วย uvicorn)
//    - "browser" : รันโมเดล .onnx ในเบราว์เซอร์เลย (infer.js) ตอนเว็บอยู่บน hosting แบบ static
//  ทั้งสองวิธีคืน JSON หน้าตาเดียวกัน -> โค้ดวาดผลใช้ร่วมกันได้
// =====================================================================

const API = window.API_BASE || "";            // มาจาก config.js

// สีประจำแต่ละคลาส (ใช้ทั้งกรอบ, polygon และป้ายชื่อ)
const COLORS = {
  airplane:   "#2f7de1",
  boat:       "#11a59a",
  car:        "#e8582d",
  motorcycle: "#a64fd6",
};
// ชื่อภาษาไทยไว้แสดงคู่กับชื่ออังกฤษ
const THAI = { airplane: "เครื่องบิน", boat: "เรือ", car: "รถยนต์", motorcycle: "มอเตอร์ไซค์" };
const MODE_LABEL = { rect: "Rectangle (detect)", poly: "Polygon (segment)" };

// ดึง element จากหน้า HTML มาเก็บในตัวแปร
const $ = (id) => document.getElementById(id);
const fileInput = $("file"), drop = $("drop"), runBtn = $("run");
const confInput = $("conf"), confVal = $("confVal"), results = $("results");

let currentFile = null;    // ไฟล์ภาพที่เลือกอยู่ (Blob/File)
let currentImg = null;     // <img> ของภาพนั้น (ไว้วาดลง canvas)
let engine = "api";        // "api" หรือ "browser" (ตัดสินใจใน checkHealth)

// ---------------------------------------------------------------
// 1) เช็คสถานะ API
// ---------------------------------------------------------------
async function checkHealth() {
  const pill = $("status");
  const mode = window.INFERENCE || "auto";
  if (mode !== "browser") {
    try {
      const r = await fetch(`${API}/api/health`);
      const j = await r.json();
      if (!j.models || !j.models.length) throw new Error("no models");
      engine = "api";
      pill.textContent = `API online · models: ${j.models.join(", ")}`;
      pill.className = "pill ok";
      return;
    } catch (e) {
      if (mode === "api") {
        pill.textContent = "API offline";
        pill.className = "pill bad";
        return;
      }
    }
  }
  // ไม่มีหลังบ้าน -> รันในเบราว์เซอร์
  engine = "browser";
  pill.textContent = "Browser mode · โมเดลรันในเบราว์เซอร์ (ONNX)";
  pill.className = "pill ok";
}

// ---------------------------------------------------------------
// 2) รับภาพจากผู้ใช้
// ---------------------------------------------------------------
function setImage(file) {
  if (!file || !file.type.startsWith("image/")) return;
  currentFile = file;
  const img = new Image();
  img.onload = () => {
    currentImg = img;
    runBtn.disabled = false;
    // แสดงภาพตัวอย่างก่อนทำนาย
    results.innerHTML = "";
    results.appendChild(makeCard("ภาพที่เลือก", null, img));
  };
  img.src = URL.createObjectURL(file);
}

// เลือกไฟล์ผ่านปุ่ม
fileInput.addEventListener("change", (e) => setImage(e.target.files[0]));

// ลากไฟล์มาวาง (drag & drop)
["dragenter", "dragover"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("hover"); }));
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("hover"); }));
drop.addEventListener("drop", (e) => setImage(e.dataTransfer.files[0]));

// ภาพตัวอย่าง: อ่านรายชื่อจาก samples/samples.json แล้วทำเป็นปุ่มรูปเล็ก ๆ
async function loadSamples() {
  try {
    const list = await (await fetch("samples/samples.json")).json();
    const box = $("samples");
    list.forEach((name) => {
      const b = document.createElement("button");
      b.className = "thumb";
      b.title = name;
      b.innerHTML = `<img src="samples/${name}" alt="${name}" loading="lazy">`;
      b.onclick = async () => {
        // แปลงภาพตัวอย่างเป็น File เหมือนผู้ใช้อัปโหลดเอง
        const blob = await (await fetch(`samples/${name}`)).blob();
        setImage(new File([blob], name, { type: blob.type || "image/jpeg" }));
      };
      box.appendChild(b);
    });
  } catch (e) { /* ไม่มีภาพตัวอย่างก็ไม่เป็นไร */ }
}

// แสดงค่า confidence ข้าง slider
confInput.addEventListener("input", () => (confVal.textContent = (+confInput.value).toFixed(2)));

// ---------------------------------------------------------------
// 3) ส่งภาพไปให้ API ทำนาย
// ---------------------------------------------------------------
async function callApi(mode) {
  // โหมด browser: ไม่ต้องส่งไปไหน รันโมเดลในเครื่องผู้ใช้เลย (ดู infer.js)
  if (engine === "browser") return Infer.predict(currentImg, mode, +confInput.value);

  // multipart/form-data = รูปแบบเดียวกับฟอร์มอัปโหลดไฟล์
  const fd = new FormData();
  fd.append("file", currentFile);
  fd.append("mode", mode);
  fd.append("conf", confInput.value);
  const r = await fetch(`${API}/api/predict`, { method: "POST", body: fd });
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
  return r.json();
}

async function runPredict() {
  if (!currentFile) return;
  const mode = document.querySelector("input[name=mode]:checked").value;
  const modes = mode === "both" ? ["rect", "poly"] : [mode];
  runBtn.disabled = true;
  runBtn.textContent = "กำลังทำนาย…";
  results.innerHTML = "";
  try {
    // เรียก API พร้อมกัน (ถ้าเลือกเทียบทั้งสองแบบ)
    const outs = await Promise.all(modes.map(callApi));
    outs.forEach((out) => results.appendChild(renderResult(out)));
  } catch (e) {
    const err = document.createElement("div");
    err.className = "empty error";
    err.textContent = `เกิดข้อผิดพลาด: ${e.message}`;   // textContent = กันโค้ด HTML แปลกปลอม
    results.appendChild(err);
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "3. ทำนาย (Predict)";
  }
}
runBtn.addEventListener("click", runPredict);

// ---------------------------------------------------------------
// 4) วาดผลลัพธ์
// ---------------------------------------------------------------
function makeCard(title, subtitle, img) {
  // card = กล่องผลลัพธ์ 1 กล่อง (หัวข้อ + canvas + ตาราง)
  const card = document.createElement("article");
  card.className = "card";
  card.innerHTML = `<header><h2>${title}</h2>${subtitle ? `<span>${subtitle}</span>` : ""}</header>`;
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;     // ใช้ขนาดจริงของภาพ -> พิกัดจาก API ใช้ได้ตรง ๆ
  canvas.height = img.naturalHeight;
  canvas.getContext("2d").drawImage(img, 0, 0);
  card.appendChild(canvas);
  return card;
}

function renderResult(out) {
  const top = out.top_class;
  const headline = top ? `${top} · ${THAI[top] || ""}` : "ไม่พบวัตถุ";
  const card = makeCard(headline, `${MODE_LABEL[out.mode]} · ${out.count} objects · ${out.inference_ms} ms`, currentImg);
  const canvas = card.querySelector("canvas");
  const ctx = canvas.getContext("2d");

  // ปรับความหนาเส้น/ขนาดตัวอักษรตามขนาดภาพ (ภาพใหญ่ เส้นก็หนาขึ้น)
  const s = Math.max(canvas.width, canvas.height) / 640;
  ctx.lineWidth = 3 * s;
  ctx.font = `600 ${16 * s}px system-ui, sans-serif`;

  // วาดจากมั่นใจน้อย -> มาก เพื่อให้ตัวที่มั่นใจที่สุดอยู่บนสุด
  [...out.detections].reverse().forEach((d) => {
    const color = COLORS[d.class_name] || "#ffcc00";
    const [x1, y1, x2, y2] = d.box;

    if (d.mask) {
      // ---- Mask (มาจากโหมด browser): ย้อมสี mask แล้วยืดวาดทับภาพ ----
      const tint = document.createElement("canvas");
      tint.width = d.mask.canvas.width;
      tint.height = d.mask.canvas.height;
      const tctx = tint.getContext("2d");
      tctx.drawImage(d.mask.canvas, 0, 0);
      tctx.globalCompositeOperation = "source-in";   // ระบายสีเฉพาะตรงที่เป็น mask
      tctx.fillStyle = color;
      tctx.fillRect(0, 0, tint.width, tint.height);
      ctx.save();
      ctx.globalAlpha = 0.45;
      ctx.imageSmoothingEnabled = true;              // ขยายแล้วขอบเนียน
      ctx.drawImage(tint, ...d.mask.src, 0, 0, canvas.width, canvas.height);
      ctx.restore();
      ctx.strokeStyle = color;
      ctx.setLineDash([8 * s, 6 * s]);               // กรอบเส้นประบอกขอบเขต
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([]);
    } else if (d.polygon && d.polygon.length > 2) {
      // ---- Polygon: ระบายสีโปร่งแสง + เส้นขอบ ----
      ctx.beginPath();
      d.polygon.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
      ctx.fillStyle = color + "55";          // "55" = ความโปร่งใส ~33%
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.stroke();
    } else {
      // ---- Rectangle: วาดกรอบสี่เหลี่ยม ----
      ctx.strokeStyle = color;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }

    // ป้ายชื่อ "car 92%" เหนือกรอบ
    const text = `${d.class_name} ${(d.confidence * 100).toFixed(0)}%`;
    const tw = ctx.measureText(text).width + 10 * s;
    const th = 22 * s;
    const ty = y1 - th < 0 ? y1 : y1 - th;  // ถ้าชิดขอบบน ให้ย้ายป้ายเข้าไปในกรอบ
    ctx.fillStyle = color;
    ctx.fillRect(x1, ty, tw, th);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, x1 + 5 * s, ty + th - 6 * s);
  });

  // ตารางรายการวัตถุ
  const rows = out.detections.map((d) => `
    <tr>
      <td><i style="background:${COLORS[d.class_name]}"></i>${d.class_name} <small>${THAI[d.class_name] || ""}</small></td>
      <td><div class="bar"><span style="width:${d.confidence * 100}%;background:${COLORS[d.class_name]}"></span></div></td>
      <td class="num">${(d.confidence * 100).toFixed(1)}%</td>
    </tr>`).join("");
  const table = document.createElement("table");
  table.innerHTML = rows || `<tr><td>ไม่พบวัตถุที่ confidence ≥ ${confInput.value}</td></tr>`;
  card.appendChild(table);
  return card;
}

// เริ่มทำงานเมื่อเปิดหน้า
// ทดสอบอัตโนมัติผ่าน URL ได้ เช่น index.html?sample=car_013.jpg&mode=both
(async () => {
  await checkHealth();
  await loadSamples();
  const q = new URLSearchParams(location.search);
  if (q.get("sample")) {
    const m = document.querySelector(`input[name=mode][value=${q.get("mode") || "rect"}]`);
    if (m) m.checked = true;
    const blob = await (await fetch(`samples/${q.get("sample")}`)).blob();
    setImage(new File([blob], q.get("sample"), { type: "image/jpeg" }));
    await new Promise((ok) => setTimeout(ok, 300));   // รอภาพโหลด
    await runPredict();
    document.title = "DONE";                         // บอกสคริปต์ทดสอบว่าเสร็จแล้ว
  }
})();
