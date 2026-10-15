import { forward, loadModel, createBars, setupCanvas } from './common.js';

const canvas = document.getElementById('canvas');
let layers = null;

const bars = createBars(document.getElementById('bars'));
bars.build(Array.from({ length: 10 }, (_, d) => `<span class="digit">${d}</span>`));

const { ctx } = setupCanvas(canvas, { lineWidth: 22, onClear: () => bars.show(null), onChange: predict });

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

function predict() {
  if (!layers) return;
  const x = preprocess();
  bars.show(x ? forward(layers, x) : null);
}

loadModel('model/weights.json').then(j => { if (j) layers = j.layers; });
