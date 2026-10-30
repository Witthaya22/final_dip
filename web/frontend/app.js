const API = window.API_BASE || "";
const MODE = "poly";

const COLORS = {
  airplane:   "#2f7de1",
  boat:       "#11a59a",
  car:        "#e8582d",
  motorcycle: "#a64fd6",
};
const THAI = { airplane: "เครื่องบิน", boat: "เรือ", car: "รถยนต์", motorcycle: "มอเตอร์ไซค์" };

const $ = (id) => document.getElementById(id);
const fileInput = $("file"), drop = $("drop"), runBtn = $("run");
const confInput = $("conf"), confVal = $("confVal"), results = $("results");

let currentFile = null;
let currentImg = null;
let engine = "api";

async function checkHealth() {
  const pill = $("status");
  const mode = window.INFERENCE || "auto";
  if (mode !== "browser") {
    try {
      const r = await fetch(`${API}/api/health`);
      const j = await r.json();
      if (!(j.models || []).includes(MODE)) throw new Error("no polygon model");
      engine = "api";
      pill.textContent = "API online · Polygon model";
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
  engine = "browser";
  pill.textContent = "Browser mode · โมเดลรันในเบราว์เซอร์ (ONNX)";
  pill.className = "pill ok";
}

function setImage(file) {
  if (!file || !file.type.startsWith("image/")) return;
  currentFile = file;
  const img = new Image();
  img.onload = () => {
    currentImg = img;
    runBtn.disabled = false;
    results.innerHTML = "";
    results.appendChild(makeCard("ภาพที่เลือก", null, img));
  };
  img.src = URL.createObjectURL(file);
}

fileInput.addEventListener("change", (e) => setImage(e.target.files[0]));

["dragenter", "dragover"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("hover"); }));
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("hover"); }));
drop.addEventListener("drop", (e) => setImage(e.dataTransfer.files[0]));

confInput.addEventListener("input", () => (confVal.textContent = (+confInput.value).toFixed(2)));

async function predict() {
  if (engine === "browser") return Infer.predict(currentImg, MODE, +confInput.value);

  const fd = new FormData();
  fd.append("file", currentFile);
  fd.append("conf", confInput.value);
  const r = await fetch(`${API}/api/predict`, { method: "POST", body: fd });
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
  return r.json();
}

async function runPredict() {
  if (!currentFile) return;
  runBtn.disabled = true;
  runBtn.textContent = "กำลังทำนาย…";
  results.innerHTML = "";
  try {
    results.appendChild(renderResult(await predict()));
  } catch (e) {
    const err = document.createElement("div");
    err.className = "empty error";
    err.textContent = `เกิดข้อผิดพลาด: ${e.message}`;
    results.appendChild(err);
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "3. ทำนาย (Predict)";
  }
}
runBtn.addEventListener("click", runPredict);

function makeCard(title, subtitle, img) {
  const card = document.createElement("article");
  card.className = "card";
  card.innerHTML = `<header><h2>${title}</h2>${subtitle ? `<span>${subtitle}</span>` : ""}</header>`;
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  canvas.getContext("2d").drawImage(img, 0, 0);
  card.appendChild(canvas);
  return card;
}

function renderResult(out) {
  const top = out.top_class;
  const headline = top ? `${top} · ${THAI[top] || ""}` : "ไม่พบวัตถุ";
  const card = makeCard(headline, `${out.count} objects · ${out.inference_ms} ms`, currentImg);
  const canvas = card.querySelector("canvas");
  const ctx = canvas.getContext("2d");

  const s = Math.max(canvas.width, canvas.height) / 640;
  ctx.lineWidth = 3 * s;
  ctx.font = `600 ${16 * s}px system-ui, sans-serif`;

  [...out.detections].reverse().forEach((d) => {
    const color = COLORS[d.class_name] || "#ffcc00";
    const [x1, y1] = d.box;

    if (d.mask) {
      const tint = document.createElement("canvas");
      tint.width = d.mask.canvas.width;
      tint.height = d.mask.canvas.height;
      const tctx = tint.getContext("2d");
      tctx.drawImage(d.mask.canvas, 0, 0);
      tctx.globalCompositeOperation = "source-in";
      tctx.fillStyle = color;
      tctx.fillRect(0, 0, tint.width, tint.height);
      ctx.save();
      ctx.globalAlpha = 0.45;
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(tint, ...d.mask.src, 0, 0, canvas.width, canvas.height);
      ctx.restore();
    } else if (d.polygon && d.polygon.length > 2) {
      ctx.beginPath();
      d.polygon.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
      ctx.fillStyle = color + "55";
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.stroke();
    }

    const text = `${d.class_name} ${(d.confidence * 100).toFixed(0)}%`;
    const tw = ctx.measureText(text).width + 10 * s;
    const th = 22 * s;
    const ty = y1 - th < 0 ? y1 : y1 - th;
    ctx.fillStyle = color;
    ctx.fillRect(x1, ty, tw, th);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, x1 + 5 * s, ty + th - 6 * s);
  });

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

(async () => {
  await checkHealth();
  const testImg = new URLSearchParams(location.search).get("img");
  if (testImg) {
    const blob = await (await fetch(testImg)).blob();
    setImage(new File([blob], testImg, { type: blob.type || "image/jpeg" }));
    await new Promise((ok) => setTimeout(ok, 300));
    await runPredict();
    document.title = "DONE";
  }
})();
