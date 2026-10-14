const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const bars = document.getElementById('bars');
let layers = null;

// --- UI de la barra lateral ---
const rows = [];
for (let d = 0; d < 10; d++) {
  const li = document.createElement('li');
  li.innerHTML = `<span class="digit">${d}</span><div class="track"><div class="fill"></div></div><span class="pct">0.0%</span>`;
  bars.appendChild(li);
  rows.push({ li, fill: li.querySelector('.fill'), pct: li.querySelector('.pct') });
}
function showProbs(p) {
  const best = p ? p.indexOf(Math.max(...p)) : -1;
  rows.forEach((r, i) => {
    const v = p ? p[i] : 0;
    r.fill.style.width = (v * 100) + '%';
    r.pct.textContent = (v * 100).toFixed(1) + '%';
    r.li.classList.toggle('top', i === best);
  });
}

// --- Dibujo ---
function clearCanvas() {
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  showProbs(null);
}
ctx.lineWidth = 22; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.strokeStyle = '#fff';
let drawing = false;
function pos(e) {
  const r = canvas.getBoundingClientRect();
  return [(e.clientX - r.left) * canvas.width / r.width, (e.clientY - r.top) * canvas.height / r.height];
}
canvas.addEventListener('pointerdown', e => {
  drawing = true; canvas.setPointerCapture(e.pointerId);
  const [x, y] = pos(e);
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 0.01, y); ctx.stroke();
  predict();
});
canvas.addEventListener('pointermove', e => {
  if (!drawing) return;
  const [x, y] = pos(e);
  ctx.lineTo(x, y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(x, y);
  predict();
});
['pointerup', 'pointercancel'].forEach(t => canvas.addEventListener(t, () => { drawing = false; }));
document.getElementById('clear').addEventListener('click', clearCanvas);

// --- Preprocesado estilo MNIST: recortar, escalar a 20x20 y centrar por centro de masa en 28x28 ---
function preprocess() {
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const W = canvas.width, H = canvas.height;
  let minX = W, minY = H, maxX = -1, maxY = -1;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (data[(y * W + x) * 4] > 20) {
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
    }
  }
  if (maxX < 0) return null;
  const bw = maxX - minX + 1, bh = maxY - minY + 1;
  const scale = 20 / Math.max(bw, bh);
  const tw = Math.max(1, Math.round(bw * scale)), th = Math.max(1, Math.round(bh * scale));
  const tmp = document.createElement('canvas'); tmp.width = tmp.height = 28;
  const t = tmp.getContext('2d');
  t.fillStyle = '#000'; t.fillRect(0, 0, 28, 28);
  t.imageSmoothingQuality = 'high';
  t.drawImage(canvas, minX, minY, bw, bh, Math.round((28 - tw) / 2), Math.round((28 - th) / 2), tw, th);
  // centro de masa -> desplazar al centro
  let img = t.getImageData(0, 0, 28, 28).data;
  let m = 0, cx = 0, cy = 0;
  for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
    const v = img[(y * 28 + x) * 4]; m += v; cx += v * x; cy += v * y;
  }
  const dx = Math.round(14 - cx / m), dy = Math.round(14 - cy / m);
  const out = new Float32Array(784);
  for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
    const sx = x - dx, sy = y - dy;
    if (sx >= 0 && sx < 28 && sy >= 0 && sy < 28) out[y * 28 + x] = img[(sy * 28 + sx) * 4] / 255;
  }
  return out;
}

// --- Inferencia: MLP (ReLU) + softmax ---
function forward(x) {
  let a = x;
  layers.forEach((L, li) => {
    const n = L.b.length, z = new Float32Array(n);
    for (let j = 0; j < n; j++) z[j] = L.b[j];
    for (let i = 0; i < a.length; i++) {
      const ai = a[i]; if (ai === 0) continue;
      const w = L.W[i];
      for (let j = 0; j < n; j++) z[j] += ai * w[j];
    }
    if (li < layers.length - 1) for (let j = 0; j < n; j++) z[j] = Math.max(0, z[j]);
    a = z;
  });
  const mx = Math.max(...a);
  const e = Array.from(a, v => Math.exp(v - mx));
  const s = e.reduce((p, c) => p + c, 0);
  return e.map(v => v / s);
}
function predict() {
  if (!layers) return;
  const x = preprocess();
  showProbs(x ? forward(x) : null);
}

clearCanvas();
fetch('model/weights.json').then(r => r.json()).then(j => { layers = j.layers; })
  .catch(() => { document.querySelector('aside').insertAdjacentHTML('beforeend', '<p>No se pudo cargar model/weights.json (¿sirves la carpeta web por HTTP?)</p>'); });
