import { forward, loadModel, createBars, setupCanvas } from './common.js';
import { createGame } from './game.js';

const canvas = document.getElementById('canvas');
let layers = null;
let game = null;

const bars = createBars(document.getElementById('bars'));

// --- Dibujo (se guardan los trazos para volver a dibujarlos a 28x28) ---
let strokes = [];
const { clear } = setupCanvas(canvas, {
  lineWidth: 8,
  onStrokeStart: (x, y) => strokes.push([[x, y]]),
  onStrokePoint: (x, y) => strokes[strokes.length - 1].push([x, y]),
  onClear: () => { strokes = []; bars.show(null); },
  onChange: predict,
});

// --- Preprocesado estilo Quick, Draw!: el lado mayor ocupa ~26 px, centrado por caja, trazo de 1.5 px ---
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
  t.lineWidth = 1.5; t.lineCap = 'round'; t.lineJoin = 'round'; t.strokeStyle = '#fff';
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

function predict() {
  if (!layers) return;
  const x = preprocess();
  const p = x ? forward(layers, x) : null;
  bars.show(p);
  if (p) game?.check(p);
}

loadModel('model/weights.json').then(j => {
  if (!j) return;
  bars.build(j.labels.map(({ emoji, nombre }) => `<span class="label"><span class="emoji">${emoji}</span><span class="name">${nombre}</span></span>`));
  layers = j.layers;
  game = createGame({ labels: j.labels, clear });
});
