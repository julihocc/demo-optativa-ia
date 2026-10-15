const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const bars = document.getElementById('bars');
let layers = null;

// --- UI de la barra lateral (se crea al cargar las etiquetas del modelo) ---
const rows = [];
function buildBars(labels) {
  labels.forEach(({ emoji, nombre }) => {
    const li = document.createElement('li');
    li.innerHTML = `<span class="label"><span class="emoji">${emoji}</span>${nombre}</span><div class="track"><div class="fill"></div></div><span class="pct">0.0%</span>`;
    bars.appendChild(li);
    rows.push({ li, fill: li.querySelector('.fill'), pct: li.querySelector('.pct') });
  });
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

// --- Dibujo (se guardan los trazos para volver a dibujarlos a 28x28) ---
let strokes = [];
function clearCanvas() {
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  strokes = [];
  showProbs(null);
}
ctx.lineWidth = 8; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.strokeStyle = '#fff';
let drawing = false;
function pos(e) {
  const r = canvas.getBoundingClientRect();
  return [(e.clientX - r.left) * canvas.width / r.width, (e.clientY - r.top) * canvas.height / r.height];
}
canvas.addEventListener('pointerdown', e => {
  drawing = true; canvas.setPointerCapture(e.pointerId);
  const [x, y] = pos(e);
  strokes.push([[x, y]]);
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 0.01, y); ctx.stroke();
  predict();
});
canvas.addEventListener('pointermove', e => {
  if (!drawing) return;
  const [x, y] = pos(e);
  strokes[strokes.length - 1].push([x, y]);
  ctx.lineTo(x, y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(x, y);
  predict();
});
['pointerup', 'pointercancel'].forEach(t => canvas.addEventListener(t, () => { drawing = false; }));
document.getElementById('clear').addEventListener('click', clearCanvas);

// --- Preprocesado estilo Quick, Draw!: el lado mayor ocupa ~26 px, centrado por caja, trazo de ~2 px ---
function preprocess() {
  const pts = strokes.flat();
  if (!pts.length) return null;
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const scale = 24 / Math.max(maxX - minX, maxY - minY, 1);
  const ox = 14 - (minX + maxX) / 2 * scale, oy = 14 - (minY + maxY) / 2 * scale;
  const tmp = document.createElement('canvas'); tmp.width = tmp.height = 28;
  const t = tmp.getContext('2d');
  t.fillStyle = '#000'; t.fillRect(0, 0, 28, 28);
  t.lineWidth = 2; t.lineCap = 'round'; t.lineJoin = 'round'; t.strokeStyle = '#fff';
  strokes.forEach(s => {
    t.beginPath();
    s.forEach(([x, y], i) => (i ? t.lineTo : t.moveTo).call(t, x * scale + ox, y * scale + oy));
    if (s.length === 1) t.lineTo(s[0][0] * scale + ox + 0.01, s[0][1] * scale + oy);
    t.stroke();
  });
  const img = t.getImageData(0, 0, 28, 28).data;
  const out = new Float32Array(784);
  for (let i = 0; i < 784; i++) out[i] = img[i * 4] / 255;
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
fetch('model/weights.json').then(r => r.json()).then(j => { buildBars(j.labels); layers = j.layers; })
  .catch(() => { document.querySelector('aside').insertAdjacentHTML('beforeend', '<p>No se pudo cargar model/weights.json (¿sirves la carpeta web por HTTP?)</p>'); });
